"""Card representation for Judgement (Oh Hell). 52-card standard deck.

``card_id = suit_index * 13 + rank_index``
"""

from __future__ import annotations


class JudgementCard:
    """A single playing card."""

    suits = ["S", "H", "D", "C"]  # Spades, Hearts, Diamonds, Clubs
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.suit_index = JudgementCard.suits.index(suit)
        self.rank_index = JudgementCard.ranks.index(rank)
        self.card_id = self.suit_index * 13 + self.rank_index

    @staticmethod
    def get_deck():
        """Return a full 52-card deck."""
        return [JudgementCard(s, r) for s in JudgementCard.suits for r in JudgementCard.ranks]

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, JudgementCard):
            return self.card_id == other.card_id
        return False

    def __hash__(self):
        return hash(self.card_id)