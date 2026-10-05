"""Optional live-monitoring dashboards (extra: ``uv sync --extra dashboards``).

Imported on demand to keep the core import path free of dash/plotly.
"""

__all__ = ["app", "trends"]


def __getattr__(name):
    if name in {"app", "trends"}:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")