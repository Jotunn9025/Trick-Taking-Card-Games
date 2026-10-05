"""Round (sub-round) logic for Judgement (Oh Hell).

Handles one deal: bidding phase -> trick-taking phase.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .card import JudgementCard
from .player import JudgementPlayer
from .dealer import JudgementDealer
from .judger import JudgementJudger


class JudgementRound:
    """A single sub-round: deal, bid, play tricks."""

    # Fixed trump rotation order: Spades -> Diamonds -> Clubs -> Hearts
    TRUMP_ORDER = ["S", "D", "C", "H"]

    def __init__(
        self,
        players: List[JudgementPlayer],
        num_cards: int,
        dealer_player_id: int,
        np_random,
        round_index: int = 0,
    ):
        self.players = players
        self.num_players = len(players)
        self.num_cards = num_cards
        self.dealer_player_id = dealer_player_id
        self.np_random = np_random
        self.round_index = round_index

        # Deal cards and (possibly) reveal a trump card.
        self.dealer = JudgementDealer(np_random)
        self.dealer.new_round(players, num_cards)

        # Trump suit follows the fixed rotation.
        self.trump_suit: Optional[str] = self.TRUMP_ORDER[round_index % len(self.TRUMP_ORDER)]

        # Phase tracking
        self.is_bidding: bool = True
        self.bids_made: int = 0
        self.current_player_id: int = (dealer_player_id + 1) % self.num_players

        # Trick-taking state
        self.current_trick: List[Tuple[int, JudgementCard]] = []
        self.tricks_played: int = 0
        self.lead_player_id: int = 0  # set when trick-taking begins
        self.trick_history: List[List[Tuple[int, JudgementCard]]] = []

        # Dense reward accumulator per player (for the current round)
        self.dense_rewards: List[float] = [0.0] * self.num_players

        # All cards played so far (state extraction / determinization)
        self.played_cards: List[Tuple[int, JudgementCard]] = []

    def get_trump_card(self) -> Optional[JudgementCard]:
        return self.dealer.trump_card

    def is_over(self) -> bool:
        """Round is over when all tricks are played."""
        if self.is_bidding:
            return False
        return self.tricks_played >= self.num_cards

    def get_legal_actions(self) -> List[int]:
        """Get legal action IDs for the current player."""
        player = self.players[self.current_player_id]

        if self.is_bidding:
            is_dealer = self.current_player_id == self.dealer_player_id
            return JudgementJudger.get_legal_bid_actions(
                player, self.players, self.num_cards, is_dealer
            )
        else:
            lead_suit = self.current_trick[0][1].suit if self.current_trick else None
            return JudgementJudger.get_legal_play_actions(player, lead_suit)

    def step(self, action_id: int) -> Optional[float]:
        """
        Execute an action. Returns the dense reward for the acting player
        (``None`` during the bidding phase). Advances ``current_player_id``.
        """
        player = self.players[self.current_player_id]

        if self.is_bidding:
            self._step_bid(action_id, player)
            return None
        return self._step_play(action_id, player)

    def _step_bid(self, action_id: int, player: JudgementPlayer):
        """Process a bid action and add a hand-strength heuristic reward."""
        bid_value = JudgementJudger.action_id_to_bid(action_id)
        player.bid = bid_value
        self.bids_made += 1

        # Hand-strength heuristic: estimate expected tricks from the hand.
        expected_tricks = self._estimate_tricks(player.hand)

        # Shape the bid reward on absolute deviation from expected tricks.
        diff = abs(bid_value - expected_tricks)
        bid_reward = max(-1.0, 0.5 - (0.3 * diff))
        self.dense_rewards[player.player_id] += bid_reward

        if self.bids_made >= self.num_players:
            self.is_bidding = False
            self.lead_player_id = (self.dealer_player_id + 1) % self.num_players
            self.current_player_id = self.lead_player_id
        else:
            self.current_player_id = (self.current_player_id + 1) % self.num_players

    def _step_play(self, action_id: int, player: JudgementPlayer) -> float:
        """Process a play-card action. Returns the acting player's dense reward."""
        card_id = JudgementJudger.action_id_to_card_id(action_id)

        card = None
        for c in player.hand:
            if c.card_id == card_id:
                card = c
                break

        if card is None:
            raise ValueError(f"Card id {card_id} not in {player}'s hand")

        player.remove_card_from_hand(card)
        self.current_trick.append((player.player_id, card))
        self.played_cards.append((player.player_id, card))

        dense_reward = 0.0

        if len(self.current_trick) >= self.num_players:
            winner_id = JudgementJudger.judge_trick(self.current_trick, self.trump_suit)
            self.players[winner_id].tricks_won += 1
            self.trick_history.append(self.current_trick.copy())
            self.tricks_played += 1

            # Dense rewards for ALL players in this trick.
            for pid in range(self.num_players):
                p = self.players[pid]
                reward = JudgementJudger.compute_dense_trick_reward(p, pid == winner_id)
                self.dense_rewards[pid] += reward

            # Dense reward for the player who just completed the trick.
            dense_reward = JudgementJudger.compute_dense_trick_reward(
                player, player.player_id == winner_id
            )

            # Start the next trick with the winner leading.
            self.current_trick = []
            self.lead_player_id = winner_id
            self.current_player_id = winner_id
        else:
            self.current_player_id = (self.current_player_id + 1) % self.num_players

        return dense_reward

    # ------------------------------------------------------------------ heuristics

    def _estimate_tricks(self, hand: List[JudgementCard]) -> float:
        """
        Expected tricks estimate from hand strength (shared with MCTS heuristics).

        Trump cards are worth far more than off-suit cards; aces/kings dominate.
        """
        expected = 0.0
        for c in hand:
            if c.suit == self.trump_suit:
                if c.rank_index >= 12:
                    expected += 1.0     # Ace of trump
                elif c.rank_index >= 11:
                    expected += 0.8     # King of trump
                elif c.rank_index >= 9:
                    expected += 0.5     # T, J, Q of trump
                else:
                    expected += 0.2     # low trumps
            else:
                if c.rank_index >= 12:
                    expected += 0.5     # Ace off-suit
                elif c.rank_index >= 10:
                    expected += 0.2     # Q, K off-suit
        return expected