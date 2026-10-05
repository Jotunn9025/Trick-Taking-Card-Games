"""Game engine for Judgement (Oh Hell)."""

from .card import JudgementCard
from .dealer import JudgementDealer
from .player import JudgementPlayer
from .judger import JudgementJudger
from .round import JudgementRound
from .game import JudgementGame

__all__ = [
    "JudgementCard",
    "JudgementDealer",
    "JudgementPlayer",
    "JudgementJudger",
    "JudgementRound",
    "JudgementGame",
    # rlcard-touching symbols loaded lazily (ttcg.game.env needs rlcard)
    "JudgementEnv",
    "register_judgement_env",
    "make_judgement_env",
]


def __getattr__(name):
    if name in {"JudgementEnv", "register_judgement_env", "make_judgement_env"}:
        from . import env as _env

        return getattr(_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")