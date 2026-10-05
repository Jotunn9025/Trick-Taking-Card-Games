"""Trajectory reshaping for dense-reward NFSP training (no RL dependencies).

Kept dependency-free so it can be unit-tested without importing rlcard/torch.
"""

from __future__ import annotations


def reorganize_dense(trajectories, payoffs):
    """
    Reorganize raw rlcard trajectories into per-player transitions with dense
    trick-by-trick rewards.

    Each transition's reward is the change in the player's accumulated dense
    reward between consecutive states. On the terminal step the final game
    payoff is added, so the Q-network also learns from the exact-bid
    bonus/penalty.

    Transition fed to ``agent.feed()``: [state, action, reward, next_state, done].
    """
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        # rlcard stores (state, action, next_state) triples; dense reward is a
        # function of consecutive states, hence the stride-2 pairing.
        for i in range(0, len(trajectories[player]) - 2, 2):
            state = trajectories[player][i]
            next_state = trajectories[player][i + 2]
            done = i == len(trajectories[player]) - 3

            curr_dense = state.get("dense_rewards", [0.0] * num_players)[player]
            next_dense = next_state.get("dense_rewards", [0.0] * num_players)[player]
            reward = next_dense - curr_dense

            if done:
                reward += payoffs[player]

            transition = trajectories[player][i:i + 3].copy()
            transition.insert(2, reward)
            transition.append(done)
            new_trajectories[player].append(transition)

    return new_trajectories