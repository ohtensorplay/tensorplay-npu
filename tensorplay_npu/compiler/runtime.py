"""Offline model building, caching, and execution for the npu backend.

The captured region is exported to ONNX and handed to the Ascend Graph Engine
through the ATC offline converter; the compiled model is cached under
``~/.cache/tensorplay-ascend``.  The cache key covers the graph, the sample
shapes and dtypes, the target SoC and the settings, so a rebuild only happens
when something that could change the binary changed.

A built model runs with Ascend device buffers: host tensors are staged onto
the device for the call, and results return to whichever device the region's
inputs lived on.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Callable

from ._acl import (
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_SUCCESS,
    acl,
)
from .settings import CompilationSettings

__all__ = ["ModelPlan", "Runner", "build_plan"]

log = logging.getLogger(__name__)

# Bump when the lowering changes so models built by an older conversion
# surface are never replayed against a graph they were not lowered for.
LOWERING_ABI = "1"

#: ``atc --framework`` value naming the ONNX front end.
_ONNX_FRAMEWORK = "5"

#: ONNX tensor element types this backend can carry, keyed by the numeric
#: ``TensorProto`` value, mapped onto the TensorPlay dtype of the same width.
_ONNX_ELEMENT_TYPES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}

#: NumPy dtype carrying the same bytes as a TensorPlay dtype; ``None`` means
#: the value has no NumPy dtype of its own and is marshalled through raw
#: bytes instead.
_TP_NUMPY = {
    "float32": "float32",
    "float16": "float16",
    "float64": "float64",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "bfloat16": None,
}


def _tensorplay():
    import tensorplay

    return tensorplay


def _current_device(tp: Any) -> int:
    """Index of the NPU device the region runs on."""

    module = getattr(tp, "npu", None)
    if module is not None and hasattr(module, "current_device"):
        return int(module.current_device())
    return 0


def _numpy_kind(tp: Any, dtype: Any) -> str | None:
    """NumPy dtype name for a TensorPlay dtype, or ``None`` for bfloat16."""

    for tp_name, numpy_name in _TP_NUMPY.items():
        if getattr(tp, tp_name, None) is dtype:
            return numpy_name
    raise RuntimeError(f"the npu backend does not support tensors of {dtype}")


def _static_shape(tensor_type: Any) -> tuple[int, ...]:
    dims = []
    for dim in tensor_type.shape.dim:
        which = dim.WhichOneof("value") if hasattr(dim, "WhichOneof") else None
        if which == "dim_param":
            raise RuntimeError(
                "the npu backend builds one model per sample shape and cannot "
                "lower a region with a dynamic dimension"
            )
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _onnx_io(
    model: Any,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, tuple[int, ...]]]:
    """Read the region's I/O contract off the exported ONNX model.

    Initializers are not call inputs, so the graph's placeholder order is the
    order callers pass arguments in.  A data type this backend has no dtype
    for is reported instead of guessed.
    """

    tp = _tensorplay()
    graph = model.graph
    initializers = {value.name for value in graph.initializer}
    dtypes: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}

    def read(values: Any, what: str) -> list[str]:
        names = []
        for value in values:
            if value.name in initializers:
                continue
            info = value.type.tensor_type
            kind = int(info.elem_type)
            tp_name = _ONNX_ELEMENT_TYPES.get(kind)
            dtype = getattr(tp, tp_name, None) if tp_name else None
            if dtype is None:
                raise RuntimeError(
                    f"the npu backend does not support the data type of the "
                    f"region {what} {str(value.name)!r} (onnx element type {kind})"
                )
            name = str(value.name)
            names.append(name)
            dtypes[name] = dtype
            shapes[name] = _static_shape(info)
        return names

    input_names = read(graph.input, "input")
    output_names = read(graph.output, "output")
    return input_names, output_names, dtypes, shapes


def _graph_key(
    graph_module: Any,
    example_inputs: list[Any],
    settings: CompilationSettings,
    soc_version: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"lowering_abi={LOWERING_ABI}".encode())
    digest.update(str(graph_module.graph).encode())
    for sample in example_inputs:
        shape = tuple(int(d) for d in sample.shape)
        digest.update(f"|{shape}:{sample.dtype}".encode())
    digest.update(f"|precision={settings.precision_mode}".encode())
    digest.update(f"|soc={soc_version}".encode())
    digest.update(f"|atc_args={settings.extra_atc_args}".encode())
    return digest.hexdigest()


def _export_onnx(graph_module: Any, example_inputs: list[Any]) -> Any:
    from tensorplay import onnx as tp_onnx

    return tp_onnx.export((graph_module, *example_inputs))


def _soc_name(module: Any) -> str | None:
    """SoC the running device reports, when the runtime is reachable."""

    for holder in (module, getattr(module, "rt", None)):
        function = getattr(holder, "get_soc_name", None)
        if not callable(function):
            continue
        try:
            value = function()
        except Exception:
            continue
        if value:
            return str(value)
    return None


def _soc_version(settings: CompilationSettings) -> str:
    """SoC the offline model is built for."""

    if settings.soc_version is not None:
        return settings.soc_version
    override = os.environ.get("TENSORPLAY_NPU_SOC_VERSION") or os.environ.get(
        "ASCEND_SOC_VERSION"
    )
    if override:
        return override
    name = _soc_name(acl())
    if name:
        return name
    raise RuntimeError(
        "could not determine the Ascend SoC version; pass "
        'soc_version="Ascend910B4" (or set TENSORPLAY_NPU_SOC_VERSION)'
    )


def _atc_command(
    settings: CompilationSettings,
    onnx_path: str,
    output_base: str,
    soc_version: str,
    input_shapes: dict[str, tuple[int, ...]],
) -> list[str]:
    command = [
        settings.atc_path(),
        f"--model={onnx_path}",
        f"--framework={_ONNX_FRAMEWORK}",
        f"--output={output_base}",
        f"--soc_version={soc_version}",
        "--input_format=ND",
        f"--precision_mode={settings.precision_mode}",
        "--log=error",
    ]
    if input_shapes:
        spec = ";".join(
            f"{name}:{','.join(str(d) for d in shape)}"
            for name, shape in input_shapes.items()
        )
        command.append(f"--input_shape={spec}")
    command.extend(settings.extra_atc_args)
    return command


def _run_atc(command: list[str]) -> None:
    log.info("building an offline model with ATC")
    log.debug("atc: %s", " ".join(command))
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"the ATC offline converter was not found at {command[0]!r}; set "
            "the `atc` option (or TENSORPLAY_NPU_ATC) to its path"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"ATC failed with exit code {completed.returncode}: {detail[-2000:]}"
        )


def _compile_om(
    onnx_bytes: bytes,
    input_shapes: dict[str, tuple[int, ...]],
    settings: CompilationSettings,
    soc_version: str,
) -> bytes:
    """Run the offline converter once and return the compiled model."""

    directory = tempfile.mkdtemp(prefix="tensorplay-npu-")
    try:
        onnx_path = os.path.join(directory, "region.onnx")
        with open(onnx_path, "wb") as handle:
            handle.write(onnx_bytes)
        output_base = os.path.join(directory, "region")
        _run_atc(_atc_command(settings, onnx_path, output_base, soc_version, input_shapes))
        model_path = output_base + ".om"
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"ATC reported success but produced no model at {model_path}"
            )
        with open(model_path, "rb") as handle:
            return handle.read()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _load_or_build(
    settings: CompilationSettings, key: str, build: Callable[[], bytes]
) -> bytes:
    if settings.cache_dir is None:
        return build()
    os.makedirs(settings.cache_dir, exist_ok=True)
    path = os.path.join(settings.cache_dir, f"{key}.om")
    if os.path.exists(path):
        log.debug("offline model cache hit: %s", path)
        with open(path, "rb") as handle:
            return handle.read()
    blob = build()
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(blob)
    os.replace(tmp, path)
    return blob


class ModelPlan:
    """A compiled offline model plus the metadata execution needs."""

    def __init__(
        self,
        blob: bytes,
        input_names: list[str],
        output_names: list[str],
        io_dtypes: dict[str, Any],
        io_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        self.blob = blob
        self.input_names = input_names
        self.output_names = output_names
        self.io_dtypes = io_dtypes
        self.io_shapes = io_shapes


def build_plan(
    graph_module: Any,
    example_inputs: list[Any],
    settings: CompilationSettings,
) -> ModelPlan:
    """Export the region, compile (or load) it, and read its I/O contract."""

    model = _export_onnx(graph_module, example_inputs)
    input_names, output_names, io_dtypes, io_shapes = _onnx_io(model)
    onnx_bytes = model.SerializeToString()
    input_shapes = {name: io_shapes[name] for name in input_names}
    soc_version = _soc_version(settings)
    key = _graph_key(graph_module, example_inputs, settings, soc_version)
    blob = _load_or_build(
        settings,
        key,
        lambda: _compile_om(onnx_bytes, input_shapes, settings, soc_version),
    )
    return ModelPlan(blob, input_names, output_names, io_dtypes, io_shapes)


def _check(what: str, ret: Any) -> Any:
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{what} failed with AscendCL error code {ret}")
    return ret


def _current_context(module: Any) -> Any:
    getter = getattr(module.rt, "get_context", None)
    if callable(getter):
        try:
            value, ret = getter()
            if ret == ACL_SUCCESS:
                return value
        except Exception:
            return None
    return None


def _set_context(module: Any, context: Any) -> None:
    _check("acl.rt.set_context", module.rt.set_context(context))


def _push_context(module: Any, context: Any) -> Any:
    """Make ``context`` current and return the one that was current before."""

    previous = _current_context(module)
    _set_context(module, context)
    return previous


def _pop_context(module: Any, previous: Any) -> None:
    if not previous:
        return
    try:
        module.rt.set_context(previous)
    except Exception:
        pass


def _match(declared: list[str], actual: list[str]) -> list[int]:
    """Map each declared name onto its position in the compiled model.

    The converter usually keeps the framework's ordering, but it is free to
    reorder; matching by name keeps arbitrary argument order correct and only
    falls back to declaration order when the names do not line up.
    """

    if sorted(actual) == sorted(declared):
        return [actual.index(name) for name in declared]
    return list(range(len(declared)))


def _to_host(tp: Any, value: Any, dtype: Any, numpy_name: str | None) -> Any:
    """A host array carrying ``value`` in the byte layout the model expects."""

    import numpy as np

    tensor = value
    if isinstance(tensor, np.ndarray):
        array = tensor
    else:
        if getattr(tensor, "requires_grad", False):
            tensor = tensor.detach()
        device = getattr(tensor, "device", None)
        if device is not None and str(getattr(device, "type", device)) != "cpu":
            tensor = tensor.cpu()
        if getattr(tensor, "dtype", None) != dtype:
            tensor = tensor.to(dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        array = tensor.numpy()
    if numpy_name is None:
        # bfloat16 has no NumPy dtype; the value arrives widened to float32
        # and the high half of each float32 is the original bfloat16.
        array = np.ascontiguousarray(array, dtype=np.float32)
        array = (array.view(np.uint32) >> 16).astype(np.uint16)
    elif array.dtype != np.dtype(numpy_name):
        array = np.ascontiguousarray(array, dtype=numpy_name)
    return np.ascontiguousarray(array)


def _from_host(
    tp: Any, raw: Any, dtype: Any, shape: tuple[int, ...], numpy_name: str | None
) -> Any:
    """A TensorPlay tensor rebuilt from the bytes a model wrote."""

    import numpy as np

    if numpy_name is None:
        bits = raw.view(np.uint16)
        values = (bits.astype(np.uint32) << 16).view(np.float32)
        return tp.from_numpy(values.reshape(shape)).to(dtype)
    view = raw.view(numpy_name)
    return tp.from_numpy(view.reshape(shape))


class Runner:
    """Executable wrapper around one compiled offline model."""

    def __init__(self, plan: ModelPlan, settings: CompilationSettings) -> None:
        tp = _tensorplay()
        module = acl()

        self._plan = plan
        self._settings = settings
        self._tp = tp
        self._acl = module
        self._name = "npu"
        # Visible through the compiled region's cache so callers (and tests)
        # can tell a compiled product from an interpreter fallback.
        self._tensorplay_codegen = "npu"
        self._lock = threading.Lock()
        self._closed = False

        self._model_id = None
        self._desc = None
        self._input_dataset = None
        self._output_dataset = None
        self._input_slots: list[dict[str, Any]] = []
        self._output_slots: list[dict[str, Any]] = []
        self._context = None
        self._stream = None

        self._device = _current_device(tp)
        self._model_dir = tempfile.mkdtemp(prefix="tensorplay-npu-")
        self._model_path = os.path.join(self._model_dir, "region.om")
        with open(self._model_path, "wb") as handle:
            handle.write(plan.blob)

        _check("acl.rt.set_device", module.rt.set_device(self._device))
        # Creating a context makes it the current one, so remember what was
        # current first; every call sets this context for the duration and
        # restores the caller's afterwards.
        previous = _current_context(module)
        context, ret = module.rt.create_context(self._device)
        _check("acl.rt.create_context", ret)
        self._context = context
        stream, ret = module.rt.create_stream()
        _check("acl.rt.create_stream", ret)
        self._stream = stream

        try:
            self._load(module, plan)
        except BaseException:
            _pop_context(module, previous)
            self.close()
            raise
        else:
            _pop_context(module, previous)

    def _load(self, module: Any, plan: ModelPlan) -> None:
        model_id, ret = module.mdl.load_from_file(self._model_path)
        _check("acl.mdl.load_from_file", ret)
        self._model_id = model_id

        desc = module.mdl.create_desc()
        _check("acl.mdl.get_desc", module.mdl.get_desc(desc, model_id))
        self._desc = desc

        input_count = int(module.mdl.get_num_inputs(desc))
        output_count = int(module.mdl.get_num_outputs(desc))
        if input_count != len(plan.input_names):
            raise RuntimeError(
                f"the compiled model takes {input_count} input(s) but the "
                f"region declares {len(plan.input_names)}"
            )
        if output_count != len(plan.output_names):
            raise RuntimeError(
                f"the compiled model produces {output_count} output(s) but the "
                f"region declares {len(plan.output_names)}"
            )

        self._input_slots = [
            self._input_slot(module, desc, index) for index in range(input_count)
        ]
        self._output_slots = [
            self._output_slot(module, desc, index) for index in range(output_count)
        ]
        self._feed = _match(plan.input_names, [s["name"] for s in self._input_slots])
        self._emit = _match(plan.output_names, [s["name"] for s in self._output_slots])

        self._input_dataset = self._dataset(module, self._input_slots)
        self._output_dataset = self._dataset(module, self._output_slots)

    def _input_slot(self, module: Any, desc: Any, index: int) -> dict[str, Any]:
        size = int(module.mdl.get_input_size_by_index(desc, index))
        ptr, ret = module.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
        _check("acl.rt.malloc", ret)
        return {"name": str(module.mdl.get_input_name_by_index(desc, index)),
                "ptr": ptr, "size": size, "buffer": None}

    def _output_slot(self, module: Any, desc: Any, index: int) -> dict[str, Any]:
        size = int(module.mdl.get_output_size_by_index(desc, index))
        ptr, ret = module.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
        _check("acl.rt.malloc", ret)
        return {"name": str(module.mdl.get_output_name_by_index(desc, index)),
                "ptr": ptr, "size": size, "buffer": None}

    def _dataset(self, module: Any, slots: list[dict[str, Any]]) -> Any:
        dataset = module.mdl.create_dataset()
        for slot in slots:
            buffer = module.create_data_buffer(slot["ptr"], slot["size"])
            _, ret = module.mdl.add_dataset_buffer(dataset, buffer)
            _check("acl.mdl.add_dataset_buffer", ret)
            slot["buffer"] = buffer
        return dataset

    def __call__(self, *args: Any) -> Any:
        plan = self._plan
        if len(args) != len(plan.input_names):
            raise TypeError(
                f"{self._name} region expects {len(plan.input_names)} "
                f"input(s), got {len(args)}"
            )

        on_device = any(
            str(getattr(getattr(value, "device", None), "type", "cpu")) != "cpu"
            for value in args
        )
        target = "npu" if on_device else "cpu"

        module = self._acl
        with self._lock:
            previous = _push_context(module, self._context)
            try:
                for index, value in enumerate(args):
                    name = plan.input_names[index]
                    slot = self._input_slots[self._feed[index]]
                    array = _to_host(
                        self._tp, value, plan.io_dtypes[name], _numpy_kind(self._tp, plan.io_dtypes[name])
                    )
                    if array.nbytes != slot["size"]:
                        raise RuntimeError(
                            f"input {name!r} carries {array.nbytes} byte(s) but "
                            f"the compiled model expects {slot['size']}"
                        )
                    _check(
                        "acl.rt.memcpy",
                        module.rt.memcpy(
                            slot["ptr"],
                            slot["size"],
                            module.util.numpy_to_ptr(array),
                            array.nbytes,
                            ACL_MEMCPY_HOST_TO_DEVICE,
                        ),
                    )

                _check(
                    "acl.mdl.execute",
                    module.mdl.execute(
                        self._model_id, self._input_dataset, self._output_dataset
                    ),
                )
                _check(
                    "acl.rt.synchronize_stream",
                    module.rt.synchronize_stream(self._stream),
                )

                results = []
                for index, name in enumerate(plan.output_names):
                    slot = self._output_slots[self._emit[index]]
                    raw = _empty_bytes(slot["size"])
                    _check(
                        "acl.rt.memcpy",
                        module.rt.memcpy(
                            module.util.numpy_to_ptr(raw),
                            raw.nbytes,
                            slot["ptr"],
                            slot["size"],
                            ACL_MEMCPY_DEVICE_TO_HOST,
                        ),
                    )
                    tensor = _from_host(
                        self._tp,
                        raw,
                        plan.io_dtypes[name],
                        plan.io_shapes[name],
                        _numpy_kind(self._tp, plan.io_dtypes[name]),
                    )
                    if target != "cpu":
                        tensor = tensor.to(target)
                    results.append(tensor)
            finally:
                _pop_context(module, previous)
        return results[0] if len(results) == 1 else tuple(results)

    def close(self) -> None:
        """Release the model, its device buffers, and the context it ran in."""

        if self._closed:
            return
        self._closed = True
        module = self._acl
        previous = None
        current = False
        try:
            previous = _push_context(module, self._context)
            current = True
        except Exception:
            current = False
        try:
            for dataset, slots in (
                (self._input_dataset, self._input_slots),
                (self._output_dataset, self._output_slots),
            ):
                for slot in slots:
                    if slot.get("buffer") is not None:
                        _ignore(module.destroy_data_buffer, slot["buffer"])
                    if slot.get("ptr"):
                        _ignore(module.rt.free, slot["ptr"])
                if dataset is not None:
                    _ignore(module.mdl.destroy_dataset, dataset)
            if self._model_id is not None:
                _ignore(module.mdl.unload, self._model_id)
            if self._desc is not None:
                _ignore(module.mdl.destroy_desc, self._desc)
            if self._stream is not None:
                _ignore(module.rt.destroy_stream, self._stream)
        finally:
            if current:
                _pop_context(module, previous)
            if self._context is not None:
                _ignore(module.rt.destroy_context, self._context)
            shutil.rmtree(self._model_dir, ignore_errors=True)

    def __del__(self) -> None:  # pragma: no cover - exercised through close()
        try:
            self.close()
        except Exception:
            pass


def _empty_bytes(size: int) -> Any:
    import numpy as np

    return np.empty(size, dtype=np.uint8)


def _ignore(function: Callable[..., Any], *args: Any) -> None:
    try:
        function(*args)
    except Exception:
        pass
