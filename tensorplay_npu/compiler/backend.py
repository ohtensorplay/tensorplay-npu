"""The ``npu`` backend, selected by ``tensorplay.compile(backend="npu")``.

The captured region is lowered through the Ascend Graph Engine: the graph is
exported to ONNX, the ATC offline converter builds it into a model for the
target SoC, and the model runs on the NPU through AscendCL with device
buffers.  A region that fails to build stays on its interpreter instead of
failing the whole compile, unless ``pass_through_build_failures`` is set -- a
build failure is a coverage gap, not a user error.  Only inference regions
lower here; training regions are wrapped ahead-of-time by the compiler
frontend (forward through this backend, backward as traced).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .runtime import Runner, build_plan
from .settings import CompilationSettings

__all__ = ["backend"]

log = logging.getLogger(__name__)


def _backend(
    graph_module: Any,
    example_inputs: list[Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    try:
        import acl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The npu backend requires the Ascend CANN toolkit with its "
            "AscendCL Python bindings (`import acl`). Install the toolkit and "
            "put its python site-packages on PYTHONPATH, then retry."
        ) from exc

    settings = CompilationSettings.from_kwargs(kwargs)

    try:
        plan = build_plan(graph_module, example_inputs, settings)
    except Exception:
        if settings.pass_through_build_failures:
            raise
        log.warning(
            "offline model build failed on this region; "
            "running the region uncompiled",
            exc_info=True,
        )
        return graph_module

    return Runner(plan, settings)


try:
    from tensorplay.compiler import BackendCapabilities, declares_capabilities

    # Training regions never reach a backend directly: the frontend routes
    # them through ahead-of-time autograd before lookup, so no training
    # capability needs declaring.
    backend = declares_capabilities(
        BackendCapabilities(
            optional_deps=("acl",),
        )
    )(_backend)
except ImportError:
    # A core without the capability contract cannot read the declaration;
    # the backend stays importable and the contract gap surfaces when the
    # compiler resolves backends by name.
    backend = _backend
