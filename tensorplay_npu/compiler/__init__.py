"""Ascend graph backend for TensorPlay.

Registering through the ``tensorplay_compiler_backends`` entry-point group
makes the backend discoverable by name: with the CANN toolkit installed,
``tensorplay.compile(model, backend="npu")`` exports the captured region to
ONNX, compiles it into an offline model with the ATC converter, and runs it
on the NPU through AscendCL.
"""

from .backend import backend
from .settings import CompilationSettings

__all__ = ["backend", "CompilationSettings"]
