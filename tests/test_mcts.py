"""Tests for the pure MCTS agent (runs with numpy only — no rlcard/torch)."""

import numpy as np

from tests.conftest import make_state

from ttcg.agents.mcts import JudgementMCTSAgent, MCTSNode


class _FakeEnv:
    """Minimal env stand-in: the pure MCTS agent only needs ``env.game``."""

    def __init__(self, seed=42):
        from ttcg.game.game import JudgementGame

        self.game = JudgementGame(num_players=4, randomize_deals=False)
        self.game.np_random = np.random.RandomState(seed)


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
        assert val > 0

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
        best = root.best_child()
        assert best is not None

    def test_is_fully_expanded(self):
        root = MCTSNode()
        root.children = {0: MCTSNode(), 1: MCTSNode()}
        assert root.is_fully_expanded([0, 1]) is True
        assert root.is_fully_expanded([0, 1, 2]) is False


class TestJudgementMCTSAgent:
    def _make_agent(self, num_simulations=20, max_depth=1):
        env = _FakeEnv()
        agent = JudgementMCTSAgent(
            env=env,
            agent_player_id=0,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        return env, agent

    def test_returns_legal_action(self):
        env, agent = self._make_agent()
        env.game.init_game()
        state = make_state(env.game, 0)
        legal = list(state["legal_actions"].keys())
        action = agent.step(state)
        assert action in legal

    def test_eval_step(self):
        env, agent = self._make_agent()
        env.game.init_game()
        state = make_state(env.game, 0)
        action, info = agent.eval_step(state)
        legal = list(state["legal_actions"].keys())
        assert action in legal
        assert isinstance(info, dict)
        assert info["agent"] == "mcts"

    def test_single_legal_action(self):
        env, agent = self._make_agent()
        state = {
            "legal_actions": {5: None},
            "raw_legal_actions": [5],
            "obs": np.zeros(454, dtype=np.float32),
            "raw_obs": np.zeros(454, dtype=np.float32),
        }
        action = agent.step(state)
        assert action == 5

    def test_depth_parameter(self):
        for depth in [1, 2, 3]:
            env, agent = self._make_agent(num_simulations=10, max_depth=depth)
            assert agent.max_depth == depth
            env.game.init_game()
            state = make_state(env.game, 0)
            legal = list(state["legal_actions"].keys())
            action = agent.step(state)
            assert action in legal

    def test_estimate_tricks_empty_hand(self):
        env, agent = self._make_agent()
        assert agent._estimate_tricks([], "S") == 0.0

    def test_estimate_tricks_trump_premium(self):
        env, agent = self._make_agent()
        from ttcg.game.card import JudgementCard

        # Ace of trump (1.0) vs Ace off-suit (0.5).
        ace_trump = JudgementCard("H", "A")
        ace_off = JudgementCard("S", "A")
        assert agent._estimate_tricks([ace_trump], "H") == 1.0
        assert agent._estimate_tricks([ace_off], "H") == 0.5

    def test_terminal_scoring(self):
        env, agent = self._make_agent()
        env.game.init_game()
        while not env.game.is_over():
            legal = env.game._get_legal_actions()
            if legal:
                action = np.random.choice(legal)
                env.game.step(action)
        score = agent._score_terminal(env.game)
        assert score in [1.0, -1.0, 0.0]

    def test_full_game_with_mcts_agent(self):
        """An MCTS agent can play an entire game legally."""
        env, agent = self._make_agent(num_simulations=10, max_depth=1)
        state, pid = env.game.init_game()
        steps = 0
        while not env.game.is_over() and steps < 1000:
            state = make_state(env.game, pid)
            action = agent.step(state)
            assert action in state["legal_actions"]
            state, pid = env.game.step(action)
            steps += 1
        assert env.game.is_over()