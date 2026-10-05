"""NFSP training and evaluation for Judgement.

Wraps ``rlcard.agents.nfsp_agent.NFSPAgent`` with:

- ``reorganize_dense``: per-player transitions with dense reward deltas plus the
  terminal payoff, so the Q-network learns from dense trick-level shaping AND
  the final exact-bid score.
- Loss tracking (SL avg-policy loss + RL Q-loss) via a small monkey patch.
- CSV metric logging (payoff, RL/SL loss, bid outcome %) in the exact format
  consumed by ``ttcg/dashboards``.

Prioritized Experience Replay was evaluated on this pipeline and showed no
impact on final win rate, so the trainer uses rlcard's uniform replay memory.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np
import torch

from rlcard.agents.nfsp_agent import NFSPAgent

from .replay import reorganize_dense


# ------------------------------------------------------------------ loss tracking

def patch_agent_losses(agent):
    """Monkey-patch an NFSPAgent to track the latest SL/RL losses."""

    agent.latest_sl_loss = 0.0
    agent.latest_rl_loss = 0.0

    original_train_sl = agent.train_sl

    def tracked_train_sl(*args, orig_fn=original_train_sl, agent_ref=agent, **kwargs):
        sl_loss = orig_fn(*args, **kwargs)
        if sl_loss is not None:
            agent_ref.latest_sl_loss = sl_loss
        return sl_loss

    agent.train_sl = tracked_train_sl

    original_update = agent._rl_agent.q_estimator.update

    def tracked_update(*args, orig_fn=original_update, agent_ref=agent, **kwargs):
        rl_loss = orig_fn(*args, **kwargs)
        agent_ref.latest_rl_loss = rl_loss
        return rl_loss

    agent._rl_agent.q_estimator.update = tracked_update
    return agent


# ------------------------------------------------------------------ agent construction

def create_nfsp_agents(
    env,
    hidden_layers: Optional[List[int]] = None,
    device=None,
    rl_learning_rate: float = 0.001,
    sl_learning_rate: float = 0.005,
    anticipatory_param: float = 0.15,
    q_epsilon_decay_steps: int = 1400000,
):
    """Create NFSP agents for all players with the studied hyperparameters."""
    if hidden_layers is None:
        hidden_layers = [1024, 512, 256]
    if device is None:
        device = torch.device("cpu")

    agents = []
    for _ in range(env.num_players):
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=hidden_layers,
            reservoir_buffer_capacity=350000,  # scaled down to prevent OOM
            anticipatory_param=anticipatory_param,
            batch_size=512,
            train_every=32,
            rl_learning_rate=rl_learning_rate,
            sl_learning_rate=sl_learning_rate,
            min_buffer_size_to_learn=256,
            q_replay_memory_size=350000,  # scaled down to prevent OOM
            q_replay_memory_init_size=256,
            q_update_target_estimator_every=500,
            q_discount_factor=0.995,  # critical for terminal bid reward credit assignment
            q_epsilon_start=1.0,
            q_epsilon_end=0.05,
            q_epsilon_decay_steps=q_epsilon_decay_steps,
            q_train_every=32,
            q_mlp_layers=hidden_layers,
            evaluate_with="average_policy",
            device=device,
        )
        patch_agent_losses(agent)
        agents.append(agent)
    return agents


# ------------------------------------------------------------------ training

def train_nfsp(
    env,
    num_episodes=10000,
    evaluate_every=500,
    checkpoint_every=None,
    save_dir=None,
    verbose=True,
    agents=None,
    start_episode=0,
    rl_learning_rate=0.01,
    sl_learning_rate=0.005,
):
    """
    Train NFSP agents on the Judgement environment.

    Args:
        env: JudgementEnv instance
        num_episodes: Total training episodes
        evaluate_every: Evaluate/log metrics every N episodes
        checkpoint_every: Save checkpoints every N episodes
        save_dir: Directory to save checkpoints + training_metrics.csv
        verbose: Print progress
        agents: Optional existing agents to resume training
        start_episode: Episode number to start/resume from

    Returns:
        agents: Trained NFSP agents
    """
    if checkpoint_every is None:
        checkpoint_every = evaluate_every * 4

    if agents is None:
        decay_steps = int(num_episodes * 14 * 0.8)  # decay epsilon over 80% of the run
        agents = create_nfsp_agents(
            env,
            rl_learning_rate=rl_learning_rate,
            sl_learning_rate=sl_learning_rate,
            q_epsilon_decay_steps=decay_steps,
        )

    env.set_agents(agents)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_path = os.path.join(save_dir, "training_metrics.csv") if save_dir else None
    if csv_path is not None:
        mode = "a" if (os.path.exists(csv_path) and start_episode > 0) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                header = ["episode"]
                for pid in range(env.num_players):
                    header.extend(
                        [
                            f"player_{pid}_avg_payoff", f"player_{pid}_rl_loss",
                            f"player_{pid}_sl_loss", f"player_{pid}_won_pct",
                            f"player_{pid}_under_pct", f"player_{pid}_over_pct",
                        ]
                    )
                writer.writerow(header)

    rewards_log = []
    # Bid outcomes per episode: list of dicts [{pid: 'won'/'under'/'over'}, ...]
    outcomes_log = []

    for episode in range(start_episode + 1, num_episodes + 1):
        for agent in agents:
            agent.sample_episode_policy()

        trajectories, payoffs = env.run(is_training=True)

        ep_outcomes = {}
        for pid in range(env.num_players):
            p = env.game.players[pid]
            if p.bid is not None:
                if p.tricks_won == p.bid:
                    ep_outcomes[pid] = "won"
                elif p.tricks_won < p.bid:
                    ep_outcomes[pid] = "under"
                else:
                    ep_outcomes[pid] = "over"
            else:
                ep_outcomes[pid] = "under"  # shouldn't happen
        outcomes_log.append(ep_outcomes)

        # Custom reshape: dense trick-level rewards + terminal payoff.
        trajectories = reorganize_dense(trajectories, payoffs)
        for pid in range(env.num_players):
            for ts in trajectories[pid]:
                agents[pid].feed(ts)

        rewards_log.append(payoffs)

        if verbose and episode % evaluate_every == 0:
            avg_payoffs = np.mean(rewards_log[-evaluate_every:], axis=0)
            if start_episode > 0:
                print(f"\n\nEpisode {episode}/{num_episodes} "
                      f"(resumed from {start_episode}, {num_episodes - episode} remaining)")
            else:
                print(f"\n\nEpisode {episode}/{num_episodes}")

            window = outcomes_log[-evaluate_every:]
            outcome_pcts = {}
            for pid in range(env.num_players):
                counts = {"won": 0, "under": 0, "over": 0}
                for ep_out in window:
                    counts[ep_out[pid]] += 1
                total = len(window)
                outcome_pcts[pid] = {
                    "won": counts["won"] / total * 100,
                    "under": counts["under"] / total * 100,
                    "over": counts["over"] / total * 100,
                }

            csv_row = [episode]
            for pid in range(env.num_players):
                pcts = outcome_pcts[pid]
                print(
                    f"  Player {pid}: avg payoff = {avg_payoffs[pid]:.4f} "
                    f"| RL loss = {agents[pid].latest_rl_loss:.4f} "
                    f"| SL loss = {agents[pid].latest_sl_loss:.4f} "
                    f"| Won {pcts['won']:.1f}% Under {pcts['under']:.1f}% Over {pcts['over']:.1f}%"
                )
                csv_row.extend(
                    [
                        avg_payoffs[pid], agents[pid].latest_rl_loss,
                        agents[pid].latest_sl_loss,
                        round(pcts["won"], 2), round(pcts["under"], 2),
                        round(pcts["over"], 2),
                    ]
                )

            if csv_path is not None:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)

        if save_dir and episode % checkpoint_every == 0:
            for pid, agent in enumerate(agents):
                agent.save_checkpoint(save_dir, filename=f"nfsp_agent_{pid}_ep{episode}.pt")
            if verbose:
                print(f"  Checkpoints saved at episode {episode}")

    return agents


# ------------------------------------------------------------------ evaluation

def evaluate_agents(env, agents, num_episodes=100, logger=None):
    """Evaluate agents via greedy play (best response), for comparison runs."""
    # Force greedy evaluation: NFSP defaults to average_policy which throws
    # cards probabilistically; 'best_response' is the deterministic argmax.
    for agent in agents:
        if hasattr(agent, "evaluate_with"):
            agent.evaluate_with = "best_response"

    env.set_agents(agents)
    payoffs_sum = np.zeros(env.num_players)
    counts = [{"won": 0, "under": 0, "over": 0} for _ in range(env.num_players)]

    for _ in range(num_episodes):
        _, payoffs = env.run(is_training=False)
        if logger:
            logger.log_eval_game("pure_all", env.game.players, payoffs, env.num_players)
        payoffs_sum += payoffs

        for pid in range(env.num_players):
            p = env.game.players[pid]
            if p.bid is not None:
                if p.tricks_won == p.bid:
                    counts[pid]["won"] += 1
                elif p.tricks_won < p.bid:
                    counts[pid]["under"] += 1
                else:
                    counts[pid]["over"] += 1

    if num_episodes == 0:
        return payoffs_sum, counts

    avg_payoffs = payoffs_sum / num_episodes
    pcts = []
    for pid in range(env.num_players):
        pcts.append(
            {
                "won": counts[pid]["won"] / num_episodes * 100,
                "under": counts[pid]["under"] / num_episodes * 100,
                "over": counts[pid]["over"] / num_episodes * 100,
            }
        )
    return avg_payoffs, pcts


# ------------------------------------------------------------------ checkpoint loading

def load_nfsp_agents(env, checkpoint_dir, episode_tag, new_rl_lr=None, new_sl_lr=None):
    """Load pre-trained NFSP agents from checkpoints (CPU-forced)."""
    print(f"\n=== Loading NFSP agents from {checkpoint_dir} (episode {episode_tag}) ===")
    if new_rl_lr or new_sl_lr:
        print(f"  Updating learning rates -> RL: {new_rl_lr}, SL: {new_sl_lr}")

    agents = []
    for pid in range(env.num_players):
        if str(episode_tag) == "final":
            filename = f"nfsp_agent_{pid}_final.pt"
        else:
            filename = f"nfsp_agent_{pid}_ep{episode_tag}.pt"
        filepath = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        def to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, torch.device):
                return torch.device("cpu")
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            return obj

        checkpoint = to_cpu(checkpoint)
        if "device" in checkpoint:
            checkpoint["device"] = torch.device("cpu")

        agent = NFSPAgent.from_checkpoint(checkpoint)
        agent.policy_network.cpu()
        agent.set_device(torch.device("cpu"))

        if new_sl_lr is not None:
            agent._sl_learning_rate = new_sl_lr
            for param_group in agent.policy_network_optimizer.param_groups:
                param_group["lr"] = new_sl_lr

        if new_rl_lr is not None:
            agent._rl_agent.q_estimator.learning_rate = new_rl_lr
            for param_group in agent._rl_agent.q_estimator.optimizer.param_groups:
                param_group["lr"] = new_rl_lr

        patch_agent_losses(agent)
        agents.append(agent)
        print(f"  Loaded Player {pid} from {filename}")

    print("  All agents loaded.\n")
    return agents