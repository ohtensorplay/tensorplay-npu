"""Single import point for the AscendCL bindings.

``acl`` is the Python binding of the CANN runtime (pyACL).  It ships with the
toolkit rather than on an index, so the import is deferred to the first use
and reported as a missing optional dependency when it is absent.  The
constants below are the AscendCL enumerations this package reads while
driving an offline model.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "acl",
    "ACL_SUCCESS",
    "ACL_MEM_MALLOC_NORMAL_ONLY",
    "ACL_MEMCPY_HOST_TO_DEVICE",
    "ACL_MEMCPY_DEVICE_TO_HOST",
]

#: ``aclError`` value meaning "no error".
ACL_SUCCESS = 0
#: ``aclrtMalloc`` policy asking for ordinary device memory.
ACL_MEM_MALLOC_NORMAL_ONLY = 2
#: ``aclrtMemcpy`` direction: host buffer to device buffer.
ACL_MEMCPY_HOST_TO_DEVICE = 1
#: ``aclrtMemcpy`` direction: device buffer to host buffer.
ACL_MEMCPY_DEVICE_TO_HOST = 2

_MODULE: Any = None


def acl() -> Any:
    """Return the ``acl`` module, importing it on first use."""

    global _MODULE
    if _MODULE is None:
        import acl as runtime

        _MODULE = runtime
    return _MODULE
