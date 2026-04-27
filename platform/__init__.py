"""Platform package — path bootstrap.

Import this package (or call setup_paths()) before importing
bev_visualizer or posthoc_xai to ensure all dependencies are on sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PLATFORM_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PLATFORM_DIR.parent
_CBM_ROOT = _PROJECT_ROOT / "cbm"
_POSTHOC_ROOT = _PROJECT_ROOT / "post-hoc-xai"

# Prevent JAX from grabbing all GPU memory — must be set before jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def setup_paths() -> None:
    """Add cbm/, post-hoc-xai/, and the shared V-Max submodule to sys.path.

    Idempotent — safe to call multiple times.
    """
    candidates = [
        str(_CBM_ROOT),
        str(_POSTHOC_ROOT),
        str(_CBM_ROOT / "V-Max"),  # shared vmax package
    ]
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)


# Run on import so that `import platform.shared.contracts` works.
setup_paths()

# ---------------------------------------------------------------------------
# stdlib platform proxy
# Our package name shadows the stdlib `platform` module. Dependencies such as
# `uuid` (pulled in by matplotlib) call `platform.system()` at import time and
# will crash with AttributeError unless we expose those functions here.
# ---------------------------------------------------------------------------

def _install_stdlib_proxy() -> None:
    import importlib.util as _ilu
    import sysconfig as _sc
    try:
        _stdlib_py = Path(_sc.get_path("stdlib")) / "platform.py"
        if not _stdlib_py.exists():
            return
        _spec = _ilu.spec_from_file_location("_stdlib_platform_impl", str(_stdlib_py))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        # Inject stdlib functions into this package's namespace
        import platform as _self
        for _attr in (
            "system", "node", "release", "version", "machine",
            "processor", "python_version", "python_version_tuple",
            "python_build", "python_compiler", "python_branch",
            "python_revision", "python_implementation", "uname",
            "architecture", "platform", "java_ver", "win32_ver",
            "mac_ver", "libc_ver",
        ):
            if hasattr(_mod, _attr):
                setattr(_self, _attr, getattr(_mod, _attr))
    except Exception:
        pass


_install_stdlib_proxy()
