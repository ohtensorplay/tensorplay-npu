"""The npu graph backend: settings, the build/run bridge, and region execution.

The settings and bridge checks need nothing but this package and TensorPlay;
the bridge replaces the AscendCL bindings and the ATC converter with
stand-ins, so the path this package owns -- ONNX metadata, the converter
command, the offset cache, loading, input staging, execution and output
retrieval -- runs on any machine.  The end-to-end check needs an Ascend NPU
with the CANN toolkit and is skipped when either is missing; it fails, rather
than passing, if the region ends up running without a compiled model.
"""

import ctypes
import importlib
import json
import os
import types

import pytest

np = pytest.importorskip("numpy")
tp = pytest.importorskip("tensorplay")

from tensorplay_npu.compiler import _acl, runtime
from tensorplay_npu.compiler.settings import CompilationSettings

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1

# The I/O the stand-in converter declares, as (name, shape) pairs.
_OM_INPUTS = [("x", (8, 16)), ("w", (16, 32)), ("b", (32,))]
_OM_OUTPUTS = [("y", (8, 32))]


# -- settings ----------------------------------------------------------------

def test_precision_short_and_long_spellings_agree():
    assert CompilationSettings(precision="fp32").precision_mode == "must_keep_origin_dtype"
    assert (
        CompilationSettings(precision="must_keep_origin_dtype").precision_mode
        == "must_keep_origin_dtype"
    )
    assert CompilationSettings(precision="fp16").precision_mode == "force_fp16"
    assert CompilationSettings(precision="mix").precision_mode == "allow_mix_precision"


def test_settings_from_options_mapping():
    settings = CompilationSettings.from_kwargs(
        {"options": {"precision": "fp16", "soc_version": "Ascend910B4"}}
    )
    assert settings.precision_mode == "force_fp16"
    assert settings.soc_version == "Ascend910B4"


def test_settings_accept_frontend_keywords():
    settings = CompilationSettings.from_kwargs(
        {"mode": "default", "name": "region", "dynamic": False, "strict_native": True}
    )
    assert settings.precision == "fp32"


def test_settings_reject_unknown_option():
    with pytest.raises(TypeError):
        CompilationSettings.from_kwargs({"options": {"workspace": 1}})


def test_settings_reject_unknown_precision():
    with pytest.raises(ValueError):
        CompilationSettings(precision="int8")


def test_settings_reject_option_given_twice():
    with pytest.raises(TypeError):
        CompilationSettings.from_kwargs(
            {"atc": "/a", "options": {"atc": "/b"}}
        )


def test_extra_atc_args_are_kept_verbatim():
    settings = CompilationSettings(extra_atc_args=["--insert_op_conf=/x.cfg"])
    assert settings.extra_atc_args == ("--insert_op_conf=/x.cfg",)


def test_atc_path_prefers_the_explicit_setting_and_the_environment(monkeypatch):
    assert CompilationSettings(atc="/opt/atc").atc_path() == "/opt/atc"
    monkeypatch.setenv("TENSORPLAY_NPU_ATC", "/env/atc")
    assert CompilationSettings().atc_path() == "/env/atc"
    monkeypatch.delenv("TENSORPLAY_NPU_ATC")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/nonexistent-toolkit")
    assert CompilationSettings().atc_path().endswith("atc")


# -- the converter command ---------------------------------------------------

def test_atc_command_names_the_onnx_front_end_and_the_target():
    settings = CompilationSettings(precision="fp16", soc_version="Ascend910B4")
    command = runtime._atc_command(
        settings, "/tmp/region.onnx", "/tmp/region", "Ascend910B4",
        {"x": (8, 16), "w": (16, 32)},
    )
    assert command[0] == settings.atc_path()
    assert "--framework=5" in command
    assert "--model=/tmp/region.onnx" in command
    assert "--output=/tmp/region" in command
    assert "--soc_version=Ascend910B4" in command
    assert "--precision_mode=force_fp16" in command
    assert "--input_shape=x:8,16;w:16,32" in command


def test_atc_command_appends_extra_arguments():
    settings = CompilationSettings(extra_atc_args=["--log=info", "--op_select_implmode=high_precision"])
    command = runtime._atc_command(settings, "a.onnx", "b", "Ascend910B4", {})
    assert command[-2:] == ["--log=info", "--op_select_implmode=high_precision"]


# -- the bridge, against stand-in AscendCL and ATC ---------------------------

class _StandInAcl:
    """AscendCL surface the backend uses, backed by real host memory."""

    def __init__(self, om_inputs, om_outputs, soc_name):
        self.allocations = {}
        self.copied_bytes = []
        self.memcpy_directions = []
        self.executed = 0
        self.unloaded = []
        self.contexts = []
        self.created_contexts = 0
        self.created_streams = 0
        self._current = 0
        self.get_soc_name = lambda: soc_name
        self.rt = types.SimpleNamespace(
            set_device=lambda device: ACL_SUCCESS,
            create_context=self._create_context,
            create_stream=self._create_stream,
            get_context=lambda: (self._current, ACL_SUCCESS),
            set_context=self._set_context,
            malloc=self._malloc,
            free=self._free,
            memcpy=self._memcpy,
            synchronize_stream=lambda stream: ACL_SUCCESS,
            destroy_stream=lambda stream: ACL_SUCCESS,
            destroy_context=lambda context: ACL_SUCCESS,
        )
        self.util = types.SimpleNamespace(numpy_to_ptr=_numpy_to_ptr)
        self.mdl = types.SimpleNamespace(
            load_from_file=self._load_from_file,
            create_desc=lambda: types.SimpleNamespace(),
            get_desc=lambda desc, model_id: ACL_SUCCESS,
            get_num_inputs=lambda desc: len(self._inputs),
            get_num_outputs=lambda desc: len(self._outputs),
            get_input_name_by_index=lambda desc, i: self._inputs[i][0],
            get_output_name_by_index=lambda desc, i: self._outputs[i][0],
            get_input_size_by_index=lambda desc, i: _bytes(self._inputs[i][1]),
            get_output_size_by_index=lambda desc, i: _bytes(self._outputs[i][1]),
            create_dataset=self._create_dataset,
            add_dataset_buffer=lambda dataset, buffer: (dataset, ACL_SUCCESS),
            destroy_dataset=lambda dataset: ACL_SUCCESS,
            execute=self._execute,
            unload=self._unload,
            destroy_desc=lambda desc: ACL_SUCCESS,
        )

    # -- context / stream --

    def _create_context(self, device):
        self.created_contexts += 1
        context = self.created_contexts
        self.contexts.append(context)
        self._current = context
        return context, ACL_SUCCESS

    def _create_stream(self):
        self.created_streams += 1
        return self.created_streams, ACL_SUCCESS

    def _set_context(self, context):
        self._current = context
        return ACL_SUCCESS

    # -- memory --

    def _malloc(self, size, policy):
        buffer = ctypes.create_string_buffer(size) if size else None
        address = ctypes.addressof(buffer) if buffer else 0
        self.allocations[address] = buffer
        return address, ACL_SUCCESS

    def _free(self, pointer):
        self.allocations.pop(pointer, None)
        return ACL_SUCCESS

    def _memcpy(self, dst, dst_max, src, count, kind):
        if kind == ACL_MEMCPY_HOST_TO_DEVICE:
            self.copied_bytes.append(count)
        self.memcpy_directions.append(kind)
        if count:
            ctypes.memmove(dst, src, count)
        return ACL_SUCCESS

    # -- model --

    def _load_from_file(self, path):
        with open(path) as handle:
            meta = json.load(handle)
        self._inputs = [(name, tuple(shape)) for name, shape in meta["inputs"]]
        self._outputs = [(name, tuple(shape)) for name, shape in meta["outputs"]]
        return 1, ACL_SUCCESS

    def _execute(self, model_id, input_dataset, output_dataset):
        self.executed += 1
        return ACL_SUCCESS

    def _unload(self, model_id):
        self.unloaded.append(model_id)
        return ACL_SUCCESS

    def _create_dataset(self):
        return len(self.copied_bytes) + 100

    def create_data_buffer(self, pointer, size):
        return (pointer, size)

    def destroy_data_buffer(self, buffer):
        return ACL_SUCCESS


def _numpy_to_ptr(array):
    return array.ctypes.data


def _bytes(shape):
    total = 4
    for dim in shape:
        total *= int(dim)
    return total


class _Dim:
    def __init__(self, value):
        self.dim_value = value
        self.dim_param = ""

    def WhichOneof(self, name):
        return "dim_value"


class _Value:
    def __init__(self, name, elem_type, shape):
        self.name = name
        tensor_type = types.SimpleNamespace(
            elem_type=elem_type,
            shape=types.SimpleNamespace(dim=[_Dim(d) for d in shape]),
        )
        self.type = types.SimpleNamespace(tensor_type=tensor_type)


class _Model:
    def __init__(self, inputs, outputs):
        self.graph = types.SimpleNamespace(
            input=[_Value(n, 1, s) for n, s in inputs],
            output=[_Value(n, 1, s) for n, s in outputs],
            initializer=[],
        )

    def SerializeToString(self):
        return b"onnx-bytes"


class _Bridge:
    """Everything one bridge test needs to inspect."""

    def __init__(self, om_inputs, om_outputs, soc_name="Ascend910B4"):
        self.acl = _StandInAcl(om_inputs, om_outputs, soc_name)
        self.commands = []
        self.model = _Model(om_inputs, om_outputs)
        self.soc_name = soc_name
        # What the stand-in converter declares, which the graph may list in a
        # different order than the region's arguments.
        self.converter_inputs = om_inputs
        self.converter_outputs = om_outputs


def _install(monkeypatch, bridge):
    monkeypatch.setattr(_acl, "_MODULE", bridge.acl)
    monkeypatch.setattr(
        runtime, "_export_onnx", lambda graph_module, example_inputs: bridge.model
    )
    monkeypatch.setattr(
        runtime, "_run_atc", lambda command: _fake_atc(bridge, command)
    )


def _fake_atc(bridge, command):
    bridge.commands.append(list(command))
    base = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
    with open(base + ".om", "w") as handle:
        json.dump(
            {
                "inputs": [[name, list(shape)] for name, shape in bridge.converter_inputs],
                "outputs": [[name, list(shape)] for name, shape in bridge.converter_outputs],
            },
            handle,
        )


def _region_inputs():
    return [
        types.SimpleNamespace(shape=(8, 16), dtype=tp.float32),
        types.SimpleNamespace(shape=(16, 32), dtype=tp.float32),
        types.SimpleNamespace(shape=(32,), dtype=tp.float32),
    ]


def _graph():
    return types.SimpleNamespace(graph="region-graph")


def test_build_plan_reads_the_region_contract_and_runs_atc(tmp_path, monkeypatch):
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    _install(monkeypatch, bridge)
    settings = CompilationSettings(cache_dir=str(tmp_path))

    plan = runtime.build_plan(_graph(), _region_inputs(), settings)

    assert plan.input_names == ["x", "w", "b"]
    assert plan.output_names == ["y"]
    assert plan.io_shapes["x"] == (8, 16)
    assert plan.io_dtypes["x"] is tp.float32
    # The SoC came off the running device; the converter command carries it.
    assert "--soc_version=Ascend910B4" in bridge.commands[0]
    assert "--framework=5" in bridge.commands[0]


def test_build_plan_reuses_the_cached_model(tmp_path, monkeypatch):
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    _install(monkeypatch, bridge)
    settings = CompilationSettings(cache_dir=str(tmp_path))

    runtime.build_plan(_graph(), _region_inputs(), settings)
    runtime.build_plan(_graph(), _region_inputs(), settings)

    assert len(bridge.commands) == 1  # the second call read the cache
    assert len(list(tmp_path.glob("*.om"))) == 1


def test_runner_stages_inputs_and_returns_model_outputs(tmp_path, monkeypatch):
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    _install(monkeypatch, bridge)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    plan = runtime.build_plan(_graph(), _region_inputs(), settings)

    runner = runtime.Runner(plan, settings)
    try:
        x = tp.randn(8, 16)
        w = tp.randn(16, 32)
        b = tp.randn(32)
        out = runner(x, w, b)

        # One host-to-device copy per argument, in argument order.
        assert bridge.acl.copied_bytes == [4 * 8 * 16, 4 * 16 * 32, 4 * 32]
        assert bridge.acl.executed == 1
        assert tuple(out.shape) == (8, 32)
        assert out.dtype == tp.float32
        assert out.device.type == "cpu"  # inputs were on the host
    finally:
        runner.close()

    assert bridge.acl.unloaded == [1]
    assert bridge.acl.allocations == {}  # device buffers were freed


def test_runner_matches_arguments_to_a_reordered_model(tmp_path, monkeypatch):
    # The converter is free to list the inputs in another order; each argument
    # must still reach the buffer that carries its name.
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    bridge.converter_inputs = [("w", (16, 32)), ("b", (32,)), ("x", (8, 16))]
    _install(monkeypatch, bridge)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    plan = runtime.build_plan(_graph(), _region_inputs(), settings)

    runner = runtime.Runner(plan, settings)
    try:
        runner(tp.randn(8, 16), tp.randn(16, 32), tp.randn(32))
        assert bridge.acl.copied_bytes == [4 * 8 * 16, 4 * 16 * 32, 4 * 32]
    finally:
        runner.close()


def test_runner_rejects_the_wrong_number_of_inputs(tmp_path, monkeypatch):
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    _install(monkeypatch, bridge)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    plan = runtime.build_plan(_graph(), _region_inputs(), settings)

    runner = runtime.Runner(plan, settings)
    try:
        with pytest.raises(TypeError):
            runner(tp.randn(8, 16))
    finally:
        runner.close()


def test_backend_falls_back_when_the_converter_is_unavailable(tmp_path, monkeypatch):
    bridge = _Bridge(_OM_INPUTS, _OM_OUTPUTS)
    _install(monkeypatch, bridge)
    monkeypatch.setattr(
        runtime, "_run_atc", lambda command: (_ for _ in ()).throw(RuntimeError("no atc"))
    )
    monkeypatch.setitem(__import__("sys").modules, "acl", types.ModuleType("acl"))
    graph_module = types.SimpleNamespace(graph="region-graph")

    backend_module = importlib.import_module("tensorplay_npu.compiler.backend")
    result = backend_module._backend(
        graph_module, _region_inputs(), options={"cache_dir": str(tmp_path)}
    )
    assert result is graph_module  # the region stays uncompiled


# -- metadata reader and byte marshalling ------------------------------------

def test_bfloat16_survives_the_byte_round_trip():
    # numpy has no bfloat16 dtype, so the staging path carries it as raw
    # 16-bit halves; the value must come back bit for bit.
    source = tp.tensor([1.5, -2.25, 0.0, 3.5], dtype=tp.float32).to(tp.bfloat16)
    array = runtime._to_host(tp, source, tp.bfloat16, None)

    assert array.dtype == np.uint16
    assert array.nbytes == source.numel() * 2

    raw = np.frombuffer(array.tobytes(), dtype=np.uint8).copy()
    back = runtime._from_host(tp, raw, tp.bfloat16, tuple(source.shape), None)

    assert back.dtype == tp.bfloat16
    assert tp.equal(back, source)


def test_onnx_metadata_matches_a_real_export(tmp_path):
    pytest.importorskip("onnx")

    def region(x, w, bias):
        return tp.softmax(tp.relu(x @ w + bias), 1)

    x, w, bias = tp.randn(8, 16), tp.randn(16, 32), tp.randn(32)
    model = runtime._export_onnx(region, [x, w, bias])
    input_names, output_names, dtypes, shapes = runtime._onnx_io(model)

    assert input_names == ["x", "w", "bias"]
    assert len(output_names) == 1
    assert shapes["x"] == (8, 16)
    assert shapes["bias"] == (32,)
    assert dtypes["w"] is tp.float32


# -- end to end on a device --------------------------------------------------

def _has_acl() -> bool:
    try:
        import acl  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_acl(), reason="AscendCL bindings not installed")
def test_backend_is_discoverable_by_name():
    assert "npu" in tp.compiler.list_backends(include_unavailable=True)


@pytest.mark.skipif(not _has_acl(), reason="AscendCL bindings not installed")
def test_compiled_region_matches_eager_on_npu(tmp_path):
    tensorplay_npu = pytest.importorskip("tensorplay_npu")

    if not os.path.isfile(
        os.path.join(tensorplay_npu.ascend_home(), "include", "acl", "acl.h")
    ):
        pytest.skip("CANN toolkit not installed")
    if not os.path.isfile(CompilationSettings().atc_path()):
        pytest.skip("the ATC converter is not installed")
    tensorplay_npu.load()
    if not tp.npu.is_available():
        pytest.skip("no NPU device")

    def region(x, w, bias):
        h = tp.relu(x @ w + bias)
        return tp.softmax(h, 1)

    tp.manual_seed(0)
    x = tp.randn(8, 16, device="npu")
    w = tp.randn(16, 32, device="npu")
    bias = tp.randn(32, device="npu")

    expected = region(x, w, bias)
    # A build failure has to raise here: letting the region run eagerly would
    # compare the region against itself and pass without a compiled model.
    cache = tmp_path / "om"
    compiled = tp.compile(
        region,
        backend="npu",
        options={"cache_dir": str(cache), "pass_through_build_failures": True},
    )
    got = compiled(x, w, bias)

    assert got.device.type == "npu"
    err = (
        (got.cpu().to(tp.float64) - expected.cpu().to(tp.float64)).abs().max().item()
    )
    assert err < 1e-3, f"max error {err}"
    built = [name for name in os.listdir(cache) if name.endswith(".om")]
    assert built, f"no offline model in {cache}; the region was not compiled"
