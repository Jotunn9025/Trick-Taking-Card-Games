"""Player representation for Judgement (Oh Hell)."""

from __future__ import annotations

from typing import List, Optional

from .card import JudgementCard


class JudgementPlayer:
    """A player in the Judgement game."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[JudgementCard] = []
        self.bid: Optional[int] = None
        self.tricks_won: int = 0
        self.score: float = 0.0  # cumulative score across sub-rounds

    def reset_for_round(self):
        """Reset per-round state (hand, bid, tricks) but keep cumulative score."""
        self.hand = []
        self.bid = None
        self.tricks_won = 0

    def remove_card_from_hand(self, card: JudgementCard):
        self.hand.remove(card)

    def __str__(self):
        return f"Player_{self.player_id}"

    def __repr__(self):
        return self.__str__()