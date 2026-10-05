"""Shared pytest fixtures and src-layout path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src-layout package importable without an install step.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_state(game, player_id: int) -> dict:
    """Wrap a raw game state with rlcard-style OrderedDict legal actions."""
    from collections import OrderedDict

    state = game.get_state(player_id)
    state["legal_actions"] = OrderedDict({a: None for a in state["legal_actions"]})
    return state