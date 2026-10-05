"""Command-line entry point for the training & evaluation pipeline.

Run the 4-phase pipeline:

    Phase 1  NFSP training (or --load-checkpoint / --resume-training)
    Phase 2  All-hybrid evaluation
    Phase 3  Pure NFSP evaluation (comparison)
    Phase 4  Paired arena: Hybrid vs Pure on identical deals

Usage examples (mirroring the studied runs):

    # Train from scratch
    uv run ttcg --save-dir checkpoints --nfsp-episodes 5000 \
        --evaluate-every 500 --checkpoint-every 1000 \
        --mcts-simulations 50 --mcts-depth 2 --hybrid-games 10 \
        --eval-games 20 --hybrid-vs-pure-games 25

    # Test a trained checkpoint
    uv run ttcg --save-dir checkpoints --load-checkpoint 600000 \
        --eval-games 20 --hybrid-games 10 --hybrid-vs-pure-games 25

    # Resume training with adjusted learning rates
    uv run ttcg --save-dir checkpoints --load-checkpoint 600000 \
        --resume-training --nfsp-episodes 200000 \
        --rl-learning-rate 0.001 --sl-learning-rate 0.005 \
        --evaluate-every 500 --checkpoint-every 1000
"""

from __future__ import annotations

import argparse
import os

# Force CPU execution — the studied pipeline trains in CPU mode.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

if hasattr(torch, "cuda"):
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

from ttcg.agents.nfsp import load_nfsp_agents  # noqa: E402
from ttcg.game.env import make_judgement_env  # noqa: E402
from ttcg.logging.game_logger import GameLogger  # noqa: E402
from ttcg.pipeline.evaluate import (  # noqa: E402
    run_hybrid_evaluation,
    run_hybrid_vs_pure_evaluation,
    run_pure_nfsp_evaluation,
)
from ttcg.pipeline.train import run_nfsp_training  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ttcg — Judgement (Oh Hell) Hybrid MC-NFSP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Phase 1: training
    parser.add_argument("--nfsp-episodes", type=int, default=500,
                        help="NFSP training episodes (skipped if --load-checkpoint without --resume-training)")
    parser.add_argument("--resume-training", action="store_true",
                        help="Resume training from the loaded checkpoint instead of skipping training")
    parser.add_argument("--rl-learning-rate", type=float, default=None,
                        help="Override RL learning rate (Q-network) when resuming training")
    parser.add_argument("--sl-learning-rate", type=float, default=None,
                        help="Override SL learning rate (Average Policy) when resuming training")
    parser.add_argument("--load-checkpoint", type=str, default=None, metavar="EPISODE",
                        help="Load pre-trained NFSP from checkpoints at this episode number "
                             "(e.g. --load-checkpoint 4000 or --load-checkpoint final). "
                             "Skips training unless --resume-training is specified.")
    # Phases 2-4: evaluation
    parser.add_argument("--hybrid-games", type=int, default=10,
                        help="Hybrid MC-NFSP evaluation games (Phase 2)")
    parser.add_argument("--mcts-depth", type=int, default=2,
                        help="MCTS max depth (counts only agent own moves)")
    parser.add_argument("--mcts-simulations", type=int, default=200,
                        help="MCTS simulations per decision (recommended: 200 for depth 2, 500 for depth 3)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Pure NFSP evaluation games (Phase 3, for comparison)")
    parser.add_argument("--hybrid-vs-pure-games", type=int, default=0,
                        help="Evaluate 1 Hybrid vs 3 Pure agents (Phase 4)")
    # Logging / checkpoints
    parser.add_argument("--evaluate-every", type=int, default=None,
                        help="Evaluate every N episodes")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for per-game CSV logs")
    # Environment / algorithm
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--anticipatory-param", type=float, default=None,
                        help="NFSP anticipatory parameter (0.0=pure SL, 1.0=pure Q-learning). "
                             "Default: 0.15.")
    args = parser.parse_args(argv)

    env = make_judgement_env(seed=args.seed, num_players=4)

    print("Judgement Environment Created")
    print(f"  Players: {env.num_players}")
    print(f"  Actions: {env.num_actions}")
    print(f"  State shape: {env.state_shape}")

    logger = GameLogger(log_dir=args.log_dir)

    anticipatory_param = (
        args.anticipatory_param if args.anticipatory_param is not None else 0.15
    )
    print(f"  Anticipatory param: {anticipatory_param} "
          f"({'pure Q-learning' if anticipatory_param == 1.0 else 'pure avg-policy' if anticipatory_param == 0.0 else 'mixed'})")

    # ── Phase 1: NFSP training / loading ──
    if args.load_checkpoint is not None:
        nfsp_agents = load_nfsp_agents(
            env,
            args.save_dir,
            args.load_checkpoint,
            new_rl_lr=args.rl_learning_rate,
            new_sl_lr=args.sl_learning_rate,
        )
        if args.resume_training:
            rl_lr = args.rl_learning_rate if args.rl_learning_rate is not None else 0.001
            sl_lr = args.sl_learning_rate if args.sl_learning_rate is not None else 0.005

            start_ep = 0
            try:
                start_ep = int(args.load_checkpoint)
            except ValueError:
                pass  # 'final' or non-integer tags default to 0

            nfsp_agents = run_nfsp_training(
                env, args.nfsp_episodes, args.save_dir,
                args.evaluate_every, args.checkpoint_every,
                agents=nfsp_agents, start_episode=start_ep,
                rl_lr=rl_lr, sl_lr=sl_lr,
            )
    else:
        rl_lr = args.rl_learning_rate if args.rl_learning_rate is not None else 0.001
        sl_lr = args.sl_learning_rate if args.sl_learning_rate is not None else 0.005
        nfsp_agents = run_nfsp_training(
            env, args.nfsp_episodes, args.save_dir,
            args.evaluate_every, args.checkpoint_every,
            rl_lr=rl_lr, sl_lr=sl_lr,
        )

    # ── Phase 2: all-hybrid evaluation ──
    hybrid_avg, hybrid_pcts = run_hybrid_evaluation(
        env, nfsp_agents, args.hybrid_games,
        args.mcts_depth, args.mcts_simulations, logger=logger
    )

    # ── Phase 3: pure NFSP comparison ──
    nfsp_avg, nfsp_pcts = run_pure_nfsp_evaluation(env, nfsp_agents, args.eval_games, logger=logger)

    # ── Phase 4: paired arena ──
    hvp_results = None
    if args.hybrid_vs_pure_games > 0:
        hvp_results = run_hybrid_vs_pure_evaluation(
            env, nfsp_agents, args.hybrid_vs_pure_games,
            args.mcts_depth, args.mcts_simulations, logger=logger
        )

    print(f"\n  Game logs saved to: {args.log_dir}/")

    # ── Summary ──
    print("═" * 80)
    print("  PURE NFSP vs PURE NFSP (all players same type)")
    print("═" * 80)
    for pid in range(env.num_players):
        n_str = f'Payoff {nfsp_avg[pid]:.2f} (Won {nfsp_pcts[pid]["won"]:.1f}%)'
        print(f"  Player {pid:<3} {n_str}")

    if hvp_results is not None:
        print("═" * 80)
        print("  ARENA: HYBRID vs PURE (per seat — 1 Hybrid + 3 Pure per game)")
        print("═" * 80)
        print(f'  {"Seat":<10} {"Hybrid (1 agent)":>30} {"Pure (3 opponents avg)":>30}')
        print(f'  {"─" * 10} {"─" * 30} {"─" * 30}')
        for pid in range(env.num_players):
            s = hvp_results["per_seat"][pid]
            h_str = f'Payoff {s["h_payoff"]:.2f} (Won {s["h_win"]:.1f}%)'
            opp_str = f'Payoff {s["opp_payoff"]:.2f} (Won {s["opp_win"]:.1f}%)'
            print(f"  Seat {pid:<5} {h_str:>30} {opp_str:>30}")
        print(f'  {"─" * 10} {"─" * 30} {"─" * 30}')
        h = hvp_results["hybrid"]
        h_total_str = f'Payoff {h["payoff"]:.2f} (Won {h["win_pct"]:.1f}%)'
        opp_total_str = f'Payoff {h["opp_payoff"]:.2f} (Won {h["opp_win_pct"]:.1f}%)'
        print(f'  {"TOTAL":<10} {h_total_str:>30} {opp_total_str:>30}')
    print("═" * 80)

    if hvp_results is not None:
        print("  SEAT COMPARISON: same seat, same opponents (3 Pure NFSP)")
        print("  Hybrid (from Arena, Paired) vs Pure (from Arena, Paired)")
        print("═" * 80)
        print(f'  {"Seat":<10} {"Hybrid in seat":>30} {"Pure in seat":>30} {"Δ Win%":>10}')
        print(f'  {"─" * 10} {"─" * 30} {"─" * 30} {"─" * 10}')
        h_total_win = 0.0
        p_total_win = 0.0
        for pid in range(env.num_players):
            s = hvp_results["per_seat"][pid]
            delta = s["h_win"] - s["p_win"]
            h_str = f'Payoff {s["h_payoff"]:.2f} (Won {s["h_win"]:.1f}%)'
            p_str = f'Payoff {s["p_payoff"]:.2f} (Won {s["p_win"]:.1f}%)'
            print(f"  Seat {pid:<5} {h_str:>30} {p_str:>30} {delta:>+9.1f}%")
            h_total_win += s["h_win"]
            p_total_win += s["p_win"]
        print(f'  {"─" * 10} {"─" * 30} {"─" * 30} {"─" * 10}')
        h_avg = h_total_win / env.num_players
        p_avg = p_total_win / env.num_players
        print(f'  {"AVG":<10} {"Won " + f"{h_avg:.1f}%":>30} '
              f'{"Won " + f"{p_avg:.1f}%":>30} {h_avg - p_avg:>+9.1f}%')
        print("═" * 80)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()