"""Allow ``python -m floodsight_detection`` without eager ML imports."""

from floodsight_detection.cli import entrypoint

raise SystemExit(entrypoint())
