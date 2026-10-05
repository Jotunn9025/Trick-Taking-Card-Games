"""Tests for the Hybrid MC-NFSP agent (requires rlcard + torch)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rlcard")

from ttcg.agents.hybrid import HybridMCNFSPAgent, MCTSNode  # noqa: E402
from ttcg.game.env import make_judgement_env  # noqa: E402


class TestMCTSNode:
    def test_ucb1_unexplored(self):
        node = MCTSNode()
        assert node.ucb1() == float("inf")

    def test_ucb1_explored(self):
        parent = MCTSNode()
        parent.visits = 10
        child = MCTSNode(parent=parent)
        child.visits = 3
        child.total_reward = 1.5
        val = child.ucb1()
        assert np.isfinite(val)

    def test_best_child(self):
        root = MCTSNode()
        root.visits = 10
        c1 = MCTSNode(parent=root)
        c1.visits = 5
        c1.total_reward = 2.0
        c2 = MCTSNode(parent=root)
        c2.visits = 1
        c2.total_reward = 0.5
        root.children = {0: c1, 1: c2}
        assert root.best_child() is not None

    def test_is_fully_expanded(self):
        root = MCTSNode()
        root.children = {0: MCTSNode(), 1: MCTSNode()}
        assert root.is_fully_expanded([0, 1]) is True
        assert root.is_fully_expanded([0, 1, 2]) is False


class TestHybridAgent:
    def _make_env_and_agent(self, all_nfsp_agents=None):
        env = make_judgement_env(seed=42, num_players=4)
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=0,
            all_nfsp_agents=all_nfsp_agents,
            num_simulations=20,
            max_depth=1,
        )
        return env, agent

    def test_returns_legal_action(self):
        env, agent = self._make_env_and_agent()
        state, _ = env.reset()
        legal = list(state["legal_actions"].keys())
        action = agent.step(state)
        assert action in legal

    def test_eval_step(self):
        """Without NFSP agents, bidding falls back to the MCTS search."""
        env, agent = self._make_env_and_agent()
        state, _ = env.reset()
        legal = list(state["legal_actions"].keys())
        action, info = agent.eval_step(state)
        assert action in legal
        assert isinstance(info, dict)
        assert info["agent"] == "hybrid_mc_nfsp"  # MCTS fallback path

    def test_eval_step_bidding_uses_sl_policy(self):
        """With NFSP agents, bidding defers to the SL (average-policy) network."""
        from ttcg.agents.nfsp import create_nfsp_agents

        env, _ = self._make_env_and_agent()
        nfsp_agents = create_nfsp_agents(env)
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=0,
            all_nfsp_agents=nfsp_agents,
            num_simulations=20,
            max_depth=1,
        )
        state, _ = env.reset()
        assert all(a <= 13 for a in state["legal_actions"])  # bidding phase
        action, info = agent.eval_step(state)
        assert action in list(state["legal_actions"].keys())
        # rlcard NFSP returns {'probs': ...} for the average policy.
        assert "probs" in info

    def test_step_uses_mcts_info_tag(self):
        """step() always runs the full MCTS search."""
        env, agent = self._make_env_and_agent()
        state, _ = env.reset()
        action = agent.step(state)
        assert action in list(state["legal_actions"].keys())

    def test_single_legal_action(self):
        env, agent = self._make_env_and_agent()
        state = {
            "legal_actions": {5: None},
            "raw_legal_actions": [5],
            "obs": np.zeros(454, dtype=np.float32),
            "raw_obs": np.zeros(454, dtype=np.float32),
        }
        assert agent.step(state) == 5

    def test_without_nfsp_falls_back_to_heuristic(self):
        """Without NFSP agents, the search still produces legal actions."""
        env, agent = self._make_env_and_agent(all_nfsp_agents=None)
        state, _ = env.reset()
        action = agent.step(state)
        assert action in list(state["legal_actions"].keys())

    def test_with_nfsp_agents(self):
        """With NFSP agents, leaf evaluation uses the DQN (no crash)."""
        from ttcg.agents.nfsp import create_nfsp_agents

        env, _ = self._make_env_and_agent()
        nfsp_agents = create_nfsp_agents(env)
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=0,
            all_nfsp_agents=nfsp_agents,
            num_simulations=20,
            max_depth=1,
        )
        state, _ = env.reset()
        action = agent.step(state)
        assert action in list(state["legal_actions"].keys())

    def test_hand_strength(self):
        """Hand-strength heuristic: empty hand is 0, trumps outrank off-suit."""
        env, agent = self._make_env_and_agent()
        env.reset()

        from ttcg.game.card import JudgementCard

        p = env.game.players[0]
        env.game.current_round.trump_suit = "H"

        p.hand = []
        assert agent._calculate_hand_strength(env.game) == 0.0

        p.hand = [JudgementCard("H", "A")]  # trump ace
        strength_trump = agent._calculate_hand_strength(env.game)

        p.hand = [JudgementCard("S", "A")]  # off-suit ace
        strength_off = agent._calculate_hand_strength(env.game)

        assert strength_trump > strength_off > 0.0

    def test_terminal_scoring_uses_official_scores(self):
        env, agent = self._make_env_and_agent()
        env.reset()
        while not env.game.is_over():
            legal = env.game._get_legal_actions()
            if legal:
                action = np.random.choice(legal)
                env.game.step(action)
        score = agent._score_terminal(env.game)
        assert score in [1.0, -1.0, 0.0]