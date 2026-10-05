"""Game orchestrator for Judgement (Oh Hell). RLCard-compatible ``Game`` interface.

Manages sub-rounds. The default schedule is a single 13-card deal (the config
that the Hybrid MC-NFSP work studies): every player gets 13 of the 52 cards
and no trump card is revealed, though the round still rotates a trump suit.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .player import JudgementPlayer
from .round import JudgementRound
from .judger import JudgementJudger


class JudgementGame:
    """Top-level game: orchestrates sub-rounds and provides the RLCard API."""

    def __init__(self, allow_step_back=False, num_players=4, randomize_deals=True):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = np.random.RandomState()

        # Intentional: deals are NOT driven by the seeded game RNG by default.
        # Seeded self-play would let agents memorize deal sequences (see
        # game/dealer.py). Set randomize_deals=False to reproduce deals.
        self._randomize_deals = randomize_deals

        # Sub-round schedule: a single 13-card deal (52 // num_players).
        max_cards = 52 // num_players
        self._round_schedule = [max_cards]

        self.players: List[JudgementPlayer] = []
        self.current_round: Optional[JudgementRound] = None
        self.round_index: int = 0
        self.dealer_index: int = 0  # rotates each sub-round
        self._game_over: bool = False

    def init_game(self):
        """Start a new full game. Returns (state, current_player_id)."""
        self.players = [JudgementPlayer(i) for i in range(self.num_players)]
        self.round_index = 0
        self.dealer_index = 0
        self._game_over = False

        self._start_new_round()

        current_player_id = self.current_round.current_player_id
        state = self.get_state(current_player_id)
        return state, current_player_id

    def _start_new_round(self):
        """Initialize the next sub-round."""
        num_cards = self._round_schedule[self.round_index]
        self.current_round = JudgementRound(
            players=self.players,
            num_cards=num_cards,
            dealer_player_id=self.dealer_index,
            np_random=(None if self._randomize_deals else self.np_random),
            round_index=self.round_index,
        )

    def step(self, action):
        """
        Take a game step. ``action`` is an int action_id.
        Returns (next_state, next_player_id).
        """
        if self._game_over:
            raise ValueError("Game is already over")

        self.current_round.step(action)

        if self.current_round.is_over():
            self._finalize_round()
            if self.round_index < len(self._round_schedule):
                self._start_new_round()
            else:
                self._game_over = True

        if self._game_over:
            state = self.get_state(0)
            return state, 0

        current_player_id = self.current_round.current_player_id
        state = self.get_state(current_player_id)
        return state, current_player_id

    def _finalize_round(self):
        """Score the completed sub-round and advance."""
        scores = JudgementJudger.compute_round_scores(self.players)
        for i, s in enumerate(scores):
            self.players[i].score += s

        self.round_index += 1
        self.dealer_index = (self.dealer_index + 1) % self.num_players

    def is_over(self) -> bool:
        return self._game_over

    # ------------------------------------------------------------------ checkpointing

    def save_checkpoint(self):
        """
        Lightweight snapshot of the full game state (no deep copy).

        Cards are immutable so we only copy list references and scalars.
        Used by MCTS to avoid an expensive ``copy.deepcopy`` per simulation and
        by the paired-arena evaluation to replay the exact same deal to two
        different agent line-ups.
        """
        cp = {
            "round_index": self.round_index,
            "dealer_index": self.dealer_index,
            "_game_over": self._game_over,
            "player_states": [
                {
                    "hand": list(p.hand),
                    "bid": p.bid,
                    "tricks_won": p.tricks_won,
                    "score": p.score,
                }
                for p in self.players
            ],
        }
        if self.current_round:
            rnd = self.current_round
            cp["round"] = {
                "is_bidding": rnd.is_bidding,
                "bids_made": rnd.bids_made,
                "current_player_id": rnd.current_player_id,
                "lead_player_id": rnd.lead_player_id,
                "current_trick": list(rnd.current_trick),
                "tricks_played": rnd.tricks_played,
                "trick_history": [list(t) for t in rnd.trick_history],
                "played_cards": list(rnd.played_cards),
                "dense_rewards": list(rnd.dense_rewards),
            }
        return cp

    def restore_checkpoint(self, cp):
        """Restore game state from a saved checkpoint."""
        self.round_index = cp["round_index"]
        self.dealer_index = cp["dealer_index"]
        self._game_over = cp["_game_over"]

        for i, ps in enumerate(cp["player_states"]):
            self.players[i].hand = list(ps["hand"])
            self.players[i].bid = ps["bid"]
            self.players[i].tricks_won = ps["tricks_won"]
            self.players[i].score = ps["score"]

        if "round" in cp and self.current_round:
            rnd = self.current_round
            r = cp["round"]
            rnd.is_bidding = r["is_bidding"]
            rnd.bids_made = r["bids_made"]
            rnd.current_player_id = r["current_player_id"]
            rnd.lead_player_id = r["lead_player_id"]
            rnd.current_trick = list(r["current_trick"])
            rnd.tricks_played = r["tricks_played"]
            rnd.trick_history = [list(t) for t in r["trick_history"]]
            rnd.played_cards = list(r["played_cards"])
            rnd.dense_rewards = list(r["dense_rewards"])

    # ------------------------------------------------------------------ RLCard API

    def get_player_id(self) -> int:
        if self._game_over:
            return 0
        return self.current_round.current_player_id

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return JudgementJudger.NUM_ACTIONS  # 66

    def get_state(self, player_id: int) -> dict:
        """Raw state for a player; the Env wrapper converts it to a vector."""
        state = {
            "player_id": player_id,
            "current_player_id": self.get_player_id(),
            "hand": list(self.players[player_id].hand),
            "all_players": self.players,
            "is_bidding": self.current_round.is_bidding if self.current_round else False,
            "trump_card": self.current_round.get_trump_card() if self.current_round else None,
            "trump_suit": self.current_round.trump_suit if self.current_round else None,
            "num_cards_this_round": (
                self.current_round.num_cards if self.current_round else 0
            ),
            "current_trick": (
                list(self.current_round.current_trick) if self.current_round else []
            ),
            "tricks_played": (
                self.current_round.tricks_played if self.current_round else 0
            ),
            "round_index": self.round_index,
            "total_rounds": len(self._round_schedule),
            "played_cards": (
                list(self.current_round.played_cards) if self.current_round else []
            ),
            "legal_actions": self._get_legal_actions(),
            "game_over": self._game_over,
            "dense_rewards": (
                list(self.current_round.dense_rewards) if self.current_round
                else [0.0] * self.num_players
            ),
        }
        return state

    def _get_legal_actions(self) -> List[int]:
        """Get legal action IDs for the current player."""
        if self._game_over or self.current_round is None:
            return []
        return self.current_round.get_legal_actions()