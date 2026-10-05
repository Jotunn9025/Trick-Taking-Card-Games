"""Pipeline: Phase 1 — NFSP training wrapper."""

from __future__ import annotations

import os

from ttcg.agents.nfsp import train_nfsp
from ttcg.game.env import make_judgement_env


def run_nfsp_training(env, num_episodes, save_dir, evaluate_every=None,
                      checkpoint_every=None, agents=None, start_episode=0,
                      rl_lr=0.001, sl_lr=0.005):
    """Phase 1: train (or resume) NFSP agents with logging + checkpoints."""
    eval_freq = evaluate_every if evaluate_every else max(1, num_episodes // 10)
    ckpt_freq = checkpoint_every if checkpoint_every else eval_freq * 4

    print(f"\n=== Phase 1: NFSP Training ({num_episodes} episodes) ===")

    agents = train_nfsp(
        env,
        num_episodes=num_episodes,
        evaluate_every=eval_freq,
        checkpoint_every=ckpt_freq,
        save_dir=save_dir,
        verbose=True,
        agents=agents,
        start_episode=start_episode,
        rl_learning_rate=rl_lr,
        sl_learning_rate=sl_lr,
    )
    print("\nNFSP training complete.\n")
    return agents