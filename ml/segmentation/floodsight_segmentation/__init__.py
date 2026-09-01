"""FloodSight SegFormer training infrastructure.

The top-level package intentionally imports no ML framework. This keeps CLI
discovery and configuration/manifest validation usable before the dedicated
training environment is installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
