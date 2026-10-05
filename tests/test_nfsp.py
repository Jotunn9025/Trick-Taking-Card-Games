"""Tests for dense-reward trajectory reshaping (no RL deps required)."""

import pytest

from ttcg.agents.replay import reorganize_dense


def _trajectory(dense_vectors, actions):
    """
    Build a flat rlcard-style trajectory: [s0, a0, s1, a1, ..., sT-1].
    Each state carries the FULL dense_rewards vector for all players.
    """
    traj = []
    for i, dense in enumerate(dense_vectors[:-1]):
        traj.append({"dense_rewards": dense})
        traj.append(actions[i])
    traj.append({"dense_rewards": dense_vectors[-1]})
    return traj


class TestReorganizeDense:
    def test_reward_is_dense_delta_with_terminal_payoff(self):
        # Player 0 accumulates 0.0 -> 1.2 -> 1.4 -> 1.4 (dense), payoff +1.0.
        # Player 1 accumulates 0.0 -> 0.0 -> 1.5 -> 2.0, payoff -1.0.
        dense = [
            [0.0, 0.0],
            [1.2, 0.0],
            [1.4, 1.5],
            [1.4, 2.0],
        ]
        payoffs = [1.0, -1.0]
        trajectories = [
            _trajectory(dense, [3, 4, 5]),
            _trajectory(dense, [1, 2, 0]),
        ]

        out = reorganize_dense(trajectories, payoffs)
        assert len(out) == 2

        # Player 0: deltas 1.2, 0.2, 0.0; terminal adds payoff +1.0.
        assert len(out[0]) == 3
        assert out[0][0][2] == pytest.approx(1.2)
        assert out[0][1][2] == pytest.approx(0.2)
        assert out[0][2][2] == pytest.approx(0.0 + 1.0)
        assert out[0][0][4] is False
        assert out[0][2][4] is True

        # Player 1: deltas 0.0, 1.5, 0.5; terminal adds payoff -1.0.
        assert out[1][0][2] == pytest.approx(0.0)
        assert out[1][1][2] == pytest.approx(1.5)
        assert out[1][2][2] == pytest.approx(0.5 - 1.0)

    def test_transition_shape_and_references(self):
        dense = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.0]]
        s0, s1, s2 = {"dense_rewards": dense[0]}, {"dense_rewards": dense[1]}, {"dense_rewards": dense[2]}
        trajectories = [[s0, 7, s1, 8, s2]]
        payoffs = [0.0]
        out = reorganize_dense(trajectories, payoffs)

        t0 = out[0][0]
        assert len(t0) == 5
        assert t0[0] is s0      # state preserved by identity
        assert t0[1] == 7       # action
        assert t0[3] is s1      # next_state
        assert t0[4] is False

        t1 = out[0][1]
        assert t1[0] is s1
        assert t1[3] is s2
        assert t1[4] is True    # terminal transition

    def test_missing_dense_defaults_to_zeros(self):
        trajectories = [[{"x": 1}, 0, {"x": 2}, 1, {"x": 3}]]
        payoffs = [0.5]
        out = reorganize_dense(trajectories, payoffs)
        rewards = [t[2] for t in out[0]]
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(0.5)  # just the terminal payoff


# ----------------------------------------------------------------------
#  NFSP construction smoke tests (need rlcard + torch; skipped otherwise)
# ----------------------------------------------------------------------

def test_create_nfsp_agents():
    pytest.importorskip("torch")
    pytest.importorskip("rlcard")
    from ttcg.agents.nfsp import create_nfsp_agents
    from ttcg.game.env import make_judgement_env

    env = make_judgement_env(seed=42, num_players=4)
    agents = create_nfsp_agents(env)
    assert len(agents) == 4
    for agent in agents:
        assert hasattr(agent, "latest_sl_loss")   # patch_agent_losses applied
        assert hasattr(agent, "latest_rl_loss")
        assert agent._rl_agent.memory  # (uniform) replay buffer exists


def test_patch_agent_losses_tracking():
    pytest.importorskip("torch")
    pytest.importorskip("rlcard")
    from ttcg.agents.nfsp import create_nfsp_agents
    from ttcg.game.env import make_judgement_env

    env = make_judgement_env(seed=42, num_players=4)
    agents = create_nfsp_agents(env)
    for agent in agents:
        assert agent.latest_sl_loss == 0.0
        assert agent.latest_rl_loss == 0.0