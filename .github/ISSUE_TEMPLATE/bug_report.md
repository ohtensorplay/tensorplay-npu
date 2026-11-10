---
name: Bug report
about: Something does not work as expected on the NPU backend
title: ""
labels: bug
assignees: ""
---

## Environment

- Ascend card (`npu-smi info`, first lines)
- CANN toolkit version (`ASCEND_TOOLKIT_HOME`)
- TensorPlay version (`python -c "import tensorplay; print(tensorplay.__version__)"`)
- `tensorplay-npu` version or commit
- OS / Python version

## Reproducer

```python
import tensorplay as tp
import tensorplay_npu
tensorplay_npu.load()
# ... minimal code ...
```

## Expected behavior

## Actual behavior

## Notes
