"""Pipeline package: 4-phase train & evaluate flow."""

from . import train, evaluate
from .cli import main

__all__ = ["train", "evaluate", "main"]