"""Dealer for Judgement (Oh Hell). Handles shuffling, dealing, trump reveal."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .card import JudgementCard
from .player import JudgementPlayer


class JudgementDealer:
    """Deals cards and reveals the trump card."""

    def __init__(self, np_random=None):
        if np_random is not None:
            self.np_random = np_random
        else:
            # INTENTIONAL: use a fresh, unseeded global RNG for the deal, NOT the
            # game's seeded RandomState. Seeded self-play would let agents memorize
            # deal sequences and "accidentally" become deterministic-trained.
            # Pass a seeded RandomState explicitly to reproduce deals (debugging,
            # tests, or ablation studies that require the same deal twice).
            self.np_random = np.random.default_rng()
        self.deck: List[JudgementCard] = []
        self.trump_card: Optional[JudgementCard] = None

    def new_round(self, players: List[JudgementPlayer], num_cards: int):
        """Shuffle deck, deal ``num_cards`` to each player, reveal trump from remainder."""
        self.deck = JudgementCard.get_deck()
        self.np_random.shuffle(self.deck)

        for player in players:
            player.reset_for_round()
            for _ in range(num_cards):
                player.hand.append(self.deck.pop())

        # Reveal trump card from remaining deck (if any cards remain).
        # With the default 13-card schedule (4 players) the whole deck is dealt
        # and no trump card is revealed — the round still has a trump *suit*.
        if self.deck:
            self.trump_card = self.deck[0]
        else:
            self.trump_card = None