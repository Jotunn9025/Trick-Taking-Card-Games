"""Judger for Judgement (Oh Hell). Legal actions, trick winners, scoring.

Action space
------------
- IDs 0..13  : bid values (0 through 13)
- IDs 14..65 : play card (card_id 0..51 mapped to action_id 14..65)

Scoring
-------
Judgement is a prediction game: hitting your bid exactly is all that matters.
Exceeding the bid is just as bad as falling short, so round scores are binary
(``+1`` exact / ``-1`` miss). Dense per-trick rewards are shaped on
bid-alignment (see :func:`compute_dense_trick_reward`).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .card import JudgementCard
from .player import JudgementPlayer


class JudgementJudger:
    """Determines legal actions, trick winners, and per-round scores."""

    NUM_BID_ACTIONS = 14   # bids 0..13
    NUM_PLAY_ACTIONS = 52  # one per card
    NUM_ACTIONS = NUM_BID_ACTIONS + NUM_PLAY_ACTIONS  # 66

    # ------------------------------------------------------------------ action space

    @staticmethod
    def card_to_action_id(card: JudgementCard) -> int:
        return card.card_id + JudgementJudger.NUM_BID_ACTIONS  # 14 + card_id

    @staticmethod
    def action_id_to_bid(action_id: int) -> int:
        assert 0 <= action_id < JudgementJudger.NUM_BID_ACTIONS
        return action_id

    @staticmethod
    def action_id_to_card_id(action_id: int) -> int:
        assert JudgementJudger.NUM_BID_ACTIONS <= action_id < JudgementJudger.NUM_ACTIONS
        return action_id - JudgementJudger.NUM_BID_ACTIONS

    # ------------------------------------------------------------------ legal actions

    @staticmethod
    def get_legal_bid_actions(
        player: JudgementPlayer,
        players: List[JudgementPlayer],
        num_cards: int,
        is_dealer: bool,
    ) -> List[int]:
        """
        Return legal bid action IDs.

        - Any player can bid 0..num_cards.
        - Hook rule: the dealer (last bidder) cannot make total bids == num_cards.
        """
        all_bids = list(range(num_cards + 1))  # 0..num_cards

        if is_dealer:
            other_bids_sum = sum(p.bid for p in players if p.bid is not None)
            forbidden_bid = num_cards - other_bids_sum
            if 0 <= forbidden_bid <= num_cards:
                all_bids = [b for b in all_bids if b != forbidden_bid]

        return all_bids  # these are also the action IDs (0..num_cards)

    @staticmethod
    def get_legal_play_actions(
        player: JudgementPlayer,
        lead_suit: Optional[str],
    ) -> List[int]:
        """Must follow lead suit if possible; otherwise any card is legal."""
        hand = player.hand
        if not hand:
            return []

        if lead_suit is not None:
            suited_cards = [c for c in hand if c.suit == lead_suit]
            if suited_cards:
                return [JudgementJudger.card_to_action_id(c) for c in suited_cards]

        return [JudgementJudger.card_to_action_id(c) for c in hand]

    # ------------------------------------------------------------------ trick judging

    @staticmethod
    def judge_trick(
        trick_cards: List[Tuple[int, JudgementCard]],
        trump_suit: Optional[str],
    ) -> int:
        """
        Determine the winner of a trick. Returns the winning player_id.

        - Highest trump wins if any trump is played.
        - Otherwise the highest card of the lead suit wins.
        """
        if not trick_cards:
            raise ValueError("Empty trick")

        lead_suit = trick_cards[0][1].suit
        winner_id = trick_cards[0][0]
        winning_card = trick_cards[0][1]

        # Invariant: after every step, winning_card is either a trump or the
        # current best lead-suit card, so the winner comparisons below are exact.
        for pid, card in trick_cards[1:]:
            if trump_suit and card.suit == trump_suit:
                # Any trump beats non-trump; among trumps the highest wins.
                if winning_card.suit != trump_suit or card.rank_index > winning_card.rank_index:
                    winning_card = card
                    winner_id = pid
            elif (
                card.suit == lead_suit
                and winning_card.suit != trump_suit
                and card.rank_index > winning_card.rank_index
            ):
                # Same lead suit as the current (non-trump) winner: higher rank wins.
                winning_card = card
                winner_id = pid

        return winner_id

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def compute_round_scores(players: List[JudgementPlayer]) -> List[float]:
        """
        Compute scores for a completed round.

        Exact bid: +1.0 | miss: -1.0  (binary scoring — overshooting is as bad
        as undershooting, which is what makes bidding the strategic core).
        """
        scores = []
        for p in players:
            if p.bid is not None and p.bid == p.tricks_won:
                scores.append(1.0)
            elif p.bid is not None:
                scores.append(-1.0)
            else:
                scores.append(0.0)
        return scores

    @staticmethod
    def compute_dense_trick_reward(player: JudgementPlayer, won_trick: bool) -> float:
        """
        Dense per-trick reward based on alignment with the bid.
        Call AFTER updating ``tricks_won`` for the winning player.

        Only an exact bid match scores in Judgement, so exceeding the bid is
        just as bad as falling short:

        ============= ================= ========================================
        outcome       state             reward
        ============= ================= ========================================
        won           still need/more   +1.0  (on track to meet bid)
        won           over bid          -0.5  (won but exceeded bid)
        lost          exactly at bid    +0.5  (good: avoided over-shooting)
        lost          still need tricks -0.3  (fell further behind)
        lost          over bid           0.0  (damage already done)
        ============= ================= ========================================
        """
        if player.bid is None:
            return 0.0

        remaining_needed = player.bid - player.tricks_won

        if won_trick:
            if remaining_needed >= 0:
                return 1.0
            else:
                return -0.5
        else:
            if remaining_needed == 0:
                return 0.5
            elif remaining_needed > 0:
                return -0.3
            else:
                return 0.0