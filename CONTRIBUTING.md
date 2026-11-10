# Contributing to tensorplay-npu

Thanks for helping!  This backend is small and deliberately follows the
patterns of the TensorPlay out-of-tree device slot.  Please keep changes in
the same style so the whole file stays readable in one pass.

## Development setup

You need a machine with:

- a Huawei Ascend NPU card and the **CANN toolkit**
  (`/usr/local/Ascend/cann-8.5.0`, or set `ASCEND_TOOLKIT_HOME`);
- TensorPlay installed (`pip install tensorplay`) or a checkout on
  `PYTHONPATH`;
- Python 3.10+ and `pytest`.

```bash
pip install -e .
pytest tests
```

The tests compare every kernel against the same computation on the host and
skip cleanly when no toolkit or device is present.

## Adding or changing a kernel

- Kernels live in `tensorplay_npu/csrc/*.cpp`, grouped by area
  (`CopyKernels`, `BinaryKernels`, `UnaryKernels`, `MatmulKernels`,
  `ReduceKernels`).
- Cast a kernel to its exact signature from
  `tensorplay/ops/KernelSignatures.h` (`sig::add__Tensor`, ...) so a mismatch
  is a compile error, not undefined behaviour.
- Follow the host kernels' conventions exactly (statistics shapes and
  dtypes, mask dtypes, which total a reduction reports) so a forward on the
  device pairs with a backward on either side.
- If the device kernel does not cover an input (shape, dtype, stride), fall
  back to the host path instead of guessing.
- Add a case to `tests/test_npu.py` that checks the device result against
  the host.
- New operator-library calls need the `aclnnXxxGetWorkspaceSize` /
  `aclnnXxx` pair; add the symbol to `AscendOpApi.*` rather than resolving
  it at each call site.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `type:
subject` with `fix(scope)` / `feat(scope)` carrying a scope
(`kernels`, `build`, `docs`, `cuda`, `frontend`, `autograd`, `compiler`).
Keep the subject under 100 characters.

## Reporting issues

Include:

- the Ascend card (`npu-smi info`) and the CANN toolkit version
  (`ASCEND_TOOLKIT_HOME`);
- the TensorPlay and `tensorplay-npu` versions;
- a minimal reproducer.
