"""Compilation settings for the npu graph backend.

Only what an offline model build consumes: the arithmetic precision, the
Ascend SoC the binary targets, the ATC converter to run, and where the
compiled offline models live.  Anything left unset is discovered from the
installed toolkit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["CompilationSettings", "default_cache_dir"]

#: Precision names this package accepts, keyed by the spelling a user writes.
#: The values are the strings ATC's ``--precision_mode`` carries.
_PRECISION_MODES = {
    "fp32": "must_keep_origin_dtype",
    "must_keep_origin_dtype": "must_keep_origin_dtype",
    "fp16": "force_fp16",
    "force_fp16": "force_fp16",
    "mix": "allow_mix_precision",
    "allow_mix_precision": "allow_mix_precision",
    "allow_fp32_to_fp16": "allow_fp32_to_fp16",
}


def default_cache_dir() -> str:
    return os.environ.get(
        "TENSORPLAY_NPU_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "tensorplay-ascend"),
    )


@dataclass(frozen=True)
class CompilationSettings:
    """Immutable knobs for one offline model build."""

    #: ``"fp32"`` (keep the exported precision), ``"fp16"`` (force half) or
    #: ``"mix"`` (let the converter mix precisions).
    precision: str = "fp32"
    #: SoC the binary targets, for example ``"Ascend910B4"``.  ``None``
    #: reads it back from the running device.
    soc_version: str | None = None
    #: Path to the ATC offline converter.  ``None`` discovers it under the
    #: toolkit root.
    atc: str | None = None
    #: Directory for compiled offline models.  ``None`` disables the cache.
    cache_dir: str | None = field(default_factory=default_cache_dir)
    #: Extra ``atc`` arguments, verbatim, appended after the ones this
    #: backend writes.
    extra_atc_args: tuple[str, ...] = ()
    #: Raise on build failure instead of running the region uncompiled.
    pass_through_build_failures: bool = False

    def __post_init__(self) -> None:
        if str(self.precision) not in _PRECISION_MODES:
            raise ValueError(
                f"unsupported precision {self.precision!r}; expected one of "
                f"{sorted(_PRECISION_MODES)}"
            )
        if self.soc_version is not None:
            version = str(self.soc_version)
            if not version:
                raise ValueError("soc_version must be a non-empty string or None")
            object.__setattr__(self, "soc_version", version)
        if self.atc is not None:
            path = str(self.atc)
            if not path:
                raise ValueError("atc must be a non-empty path or None")
            object.__setattr__(self, "atc", path)
        args = tuple(str(a) for a in self.extra_atc_args)
        object.__setattr__(self, "extra_atc_args", args)

    @property
    def precision_mode(self) -> str:
        """The ``--precision_mode`` spelling of :attr:`precision`."""

        return _PRECISION_MODES[str(self.precision)]

    def atc_path(self) -> str:
        """Path of the offline converter to run.

        An explicit :attr:`atc` wins, then ``TENSORPLAY_NPU_ATC``, then the
        locations the toolkit uses.  The first candidate is returned even when
        it is missing, so the failure names a concrete default.
        """

        if self.atc is not None:
            return self.atc
        override = os.environ.get("TENSORPLAY_NPU_ATC")
        if override:
            return override
        from .. import ascend_home

        home = ascend_home()
        candidates = [
            os.path.join(home, "bin", "atc"),
            os.path.join(home, "atc", "bin", "atc"),
            "/usr/local/Ascend/ascend-toolkit/latest/bin/atc",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "CompilationSettings":
        """Build settings from ``compile(..., backend="npu", options=...)``.

        Accepts an ``options`` mapping or flat keyword spelling; the two may
        not carry the same key, and an unknown option is rejected so a typo
        never silently changes nothing.  Keywords the compiler frontend adds
        on its own (``mode``, ``name``, ``dynamic``, ``strict_native``) are
        accepted and ignored here.
        """

        option_keys = {
            "precision",
            "soc_version",
            "atc",
            "cache_dir",
            "extra_atc_args",
            "pass_through_build_failures",
        }
        frontend_keys = {"options", "mode", "name", "dynamic", "strict_native"}
        unknown = (set(kwargs) - option_keys - frontend_keys) | (
            set(kwargs.get("options") or ()) - option_keys
        )
        if unknown:
            raise TypeError(
                f"npu backend received unsupported option(s): {sorted(unknown)}"
            )
        options = dict(kwargs.get("options") or {})
        payload: dict[str, Any] = {}
        for name in option_keys:
            in_options = name in options
            in_kwargs = name in kwargs
            if in_options and in_kwargs:
                raise TypeError(f"npu backend option {name!r} given twice")
            if in_options:
                payload[name] = options.pop(name)
            elif in_kwargs:
                payload[name] = kwargs[name]
        return cls(**payload)


#: Historical spelling kept as an alias.
Settings = CompilationSettings
