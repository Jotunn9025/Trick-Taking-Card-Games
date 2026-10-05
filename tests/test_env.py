"""Tests for the RLCard environment wrapper (requires rlcard)."""

import numpy as np
import pytest

pytest.importorskip("rlcard")

from rlcard.agents.random_agent import RandomAgent  # noqa: E402

from ttcg.game.env import make_judgement_env  # noqa: E402


class TestJudgementEnv:
    def setup_method(self):
        self.env = make_judgement_env(seed=42, num_players=4)

    def test_env_init(self):
        assert self.env.num_players == 4
        assert self.env.num_actions == 66

    def test_state_shape(self):
        """Observation shape matches the declared state_shape."""
        state, pid = self.env.reset()
        obs = state["obs"]
        expected_size = self.env.state_shape[0][1]
        assert obs.shape[0] == expected_size
        assert expected_size == 454  # documented vector size

    def test_obs_layout_components(self):
        """Spot-check the concatenated observation layout."""
        from ttcg.game.card import JudgementCard

        state, pid = self.env.reset()
        obs = state["obs"]
        # Hand one-hot occupies slots 0..51.
        hand = self.env.game.players[pid].hand
        for card in hand:
            assert obs[card.card_id] == 1.0
        # Trump suit one-hot at 52..55.
        trump_suit = self.env.game.current_round.trump_suit
        suit_idx = JudgementCard.suits.index(trump_suit)
        assert obs[52 + suit_idx] == 1.0
        # Remaining hand slots are zero (only 13 cards dealt).
        assert int(obs[:52].sum()) == 13
        # is_bidding flag at its known offset: 52+4+14+42+56+208+52+14.
        assert obs[52 + 4 + 14 + 42 + 56 + 208 + 52 + 14] == 1.0

    def test_legal_actions_in_state(self):
        state, pid = self.env.reset()
        assert "legal_actions" in state
        assert len(state["legal_actions"]) > 0

    def test_step_returns_valid_state(self):
        state, pid = self.env.reset()
        legal = list(state["legal_actions"].keys())
        next_state, next_pid = self.env.step(legal[0])
        assert "obs" in next_state
        assert 0 <= next_pid < 4

    def test_run_with_random_agents(self):
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        trajectories, payoffs = self.env.run()
        assert len(payoffs) == 4
        assert all(np.isfinite(p) for p in payoffs)

    def test_payoffs_normalized(self):
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        _, payoffs = self.env.run()
        for p in payoffs:
            assert -2.0 <= p <= 2.0, f"Payoff {p} out of expected range"

    def test_multiple_runs(self):
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        for _ in range(3):
            _, payoffs = self.env.run()
            assert len(payoffs) == 4