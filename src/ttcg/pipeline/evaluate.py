"""Pipeline: Phases 2-4 — evaluation.

- Phase 2: all-hybrid evaluation (every seat uses Hybrid MC-NFSP).
- Phase 3: pure NFSP evaluation (comparison baseline).
- Phase 4: paired arena — the same deal is played once by a line-up with one
  Hybrid seat + 3 Pure NFSP opponents, and once by 4 Pure NFSP agents. Deal
  identity is preserved via game checkpoints, producing paired outcomes that
  feed the McNemar / bootstrap analysis.
"""

from __future__ import annotations

from typing import Optional

from ttcg.agents.hybrid import HybridMCNFSPAgent
from ttcg.agents.nfsp import evaluate_agents


def run_hybrid_evaluation(env, nfsp_agents, num_games, mcts_depth,
                          mcts_simulations, logger=None):
    """Phase 2: play full games where all 4 seats use Hybrid MC-NFSP."""
    print(f"=== Phase 2: Hybrid MC-NFSP Evaluation ({num_games} games) ===")
    print(f"  MCTS depth (own moves): {mcts_depth}, Simulations: {mcts_simulations}")
    print("  Leaf evaluation: NFSP average-policy network + DQN/heuristic blend\n")

    hybrid_agents = [
        HybridMCNFSPAgent(
            env=env,
            agent_player_id=pid,
            all_nfsp_agents=nfsp_agents,
            num_simulations=mcts_simulations,
            max_depth=mcts_depth,
        )
        for pid in range(env.num_players)
    ]
    env.set_agents(hybrid_agents)

    total_payoffs = [0.0] * env.num_players
    counts = [{"won": 0, "under": 0, "over": 0} for _ in range(env.num_players)]
    for game_idx in range(1, num_games + 1):
        _, payoffs = env.run(is_training=False)
        if logger:
            logger.log_eval_game("hybrid_all", env.game.players, payoffs, env.num_players)
        for pid in range(env.num_players):
            total_payoffs[pid] += payoffs[pid]
            p = env.game.players[pid]
            if p.bid is not None:
                if p.tricks_won == p.bid:
                    counts[pid]["won"] += 1
                elif p.tricks_won < p.bid:
                    counts[pid]["under"] += 1
                else:
                    counts[pid]["over"] += 1

        if game_idx % max(1, num_games // 5) == 0:
            avg = [t / game_idx for t in total_payoffs]
            print(f"  Game {game_idx}/{num_games} — avg payoffs: {[f'{a:.4f}' for a in avg]}")

    if num_games > 0:
        final_avg = [t / num_games for t in total_payoffs]
        pcts = [
            {
                "won": c["won"] / num_games * 100,
                "under": c["under"] / num_games * 100,
                "over": c["over"] / num_games * 100,
            }
            for c in counts
        ]
        print(f"\n  Final avg payoffs: {[f'{a:.4f}' for a in final_avg]}")
        for pid in range(env.num_players):
            print(f"  Player {pid}: Won {pcts[pid]['won']:.1f}% "
                  f"Under {pcts[pid]['under']:.1f}% Over {pcts[pid]['over']:.1f}%")
    else:
        final_avg = [0.0] * env.num_players
        pcts = [{"won": 0.0, "under": 0.0, "over": 0.0} for _ in range(env.num_players)]
        print("\n  Skipped (0 games requested).")
    print("  Hybrid evaluation complete.\n")
    return final_avg, pcts


def run_pure_nfsp_evaluation(env, nfsp_agents, num_games, logger=None):
    """Phase 3: pure NFSP games for comparison."""
    print(f"=== Phase 3: Pure NFSP Evaluation ({num_games} games, for comparison) ===")
    env.set_agents(nfsp_agents)
    avg, pcts = evaluate_agents(env, nfsp_agents, num_episodes=num_games, logger=logger)
    print(f"  Pure NFSP avg payoffs: {[f'{a:.4f}' for a in avg]}")
    if num_games > 0:
        for pid in range(env.num_players):
            print(f"  Player {pid}: Won {pcts[pid]['won']:.1f}% "
                  f"Under {pcts[pid]['under']:.1f}% Over {pcts[pid]['over']:.1f}%")
    print()
    return avg, pcts


def run_copied_env(env, current_state):
    """Run an RLCard environment from a starting state until done."""
    state = current_state
    player_id = env.get_player_id()
    while not env.is_over():
        action, _ = env.agents[player_id].eval_step(state)
        state, player_id = env.step(action, env.agents[player_id].use_raw)
    return env.get_payoffs()


def run_hybrid_vs_pure_evaluation(env, nfsp_agents, num_games_per_seat,
                                  mcts_depth, mcts_simulations, logger=None):
    """Phase 4: paired arena — each seat's same deal played by Hybrid and Pure."""
    total_games = num_games_per_seat * 4
    print(f"=== Phase 4: True Arena - Hybrid vs Pure "
          f"({num_games_per_seat} games per seat, {total_games} total) ===")

    hybrid_stats = {"payoff": 0.0, "won": 0, "opp_payoff": 0.0, "opp_won": 0.0}
    pure_stats = {"payoff": 0.0, "won": 0}
    seat_stats = [
        {"h_payoff": 0.0, "h_won": 0, "opp_avg_payoff": 0.0, "opp_avg_won": 0.0,
         "p_payoff": 0.0, "p_won": 0}
        for _ in range(env.num_players)
    ]

    for hybrid_pid in range(env.num_players):
        print(f"\n  --- Testing Seat P{hybrid_pid} ---")

        # Line-up A: 1 Hybrid + 3 Pure.  Line-up B: 4 Pure.
        hybrid_mixed_agents = []
        for pid in range(env.num_players):
            if pid == hybrid_pid:
                agent = HybridMCNFSPAgent(
                    env=env,
                    agent_player_id=pid,
                    all_nfsp_agents=nfsp_agents,
                    num_simulations=mcts_simulations,
                    max_depth=mcts_depth,
                )
                hybrid_mixed_agents.append(agent)
            else:
                nfsp_agents[pid].evaluate_with = "best_response"
                hybrid_mixed_agents.append(nfsp_agents[pid])

        pure_agents_only = []
        for pid in range(env.num_players):
            nfsp_agents[pid].evaluate_with = "best_response"
            pure_agents_only.append(nfsp_agents[pid])

        def _seat_winner(pid, won_tricks):
            return 1 if won_tricks else 0

        for game_idx in range(num_games_per_seat):
            env.set_agents(hybrid_mixed_agents)
            state, _ = env.reset()
            init_checkpoint = env.game.save_checkpoint()

            # 1) Play the deal with the Hybrid seat line-up.
            hybrid_payoffs = run_copied_env(env, state)
            if logger:
                logger.log_hvp_game(hybrid_pid, "hybrid", env.game.players,
                                    hybrid_payoffs, env.num_players)

            p = env.game.players[hybrid_pid]
            h_is_won = _seat_winner(hybrid_pid, p.bid is not None and p.tricks_won == p.bid)

            opp_payoffs = [hybrid_payoffs[i] for i in range(env.num_players) if i != hybrid_pid]
            opp_wins = [
                1 if (env.game.players[i].bid is not None
                      and env.game.players[i].tricks_won == env.game.players[i].bid)
                else 0
                for i in range(env.num_players) if i != hybrid_pid
            ]
            opp_avg_p = sum(opp_payoffs) / 3.0
            opp_avg_w = sum(opp_wins) / 3.0

            hybrid_stats["payoff"] += hybrid_payoffs[hybrid_pid]
            hybrid_stats["won"] += h_is_won
            hybrid_stats["opp_payoff"] += opp_avg_p
            hybrid_stats["opp_won"] += opp_avg_w

            seat_stats[hybrid_pid]["h_payoff"] += hybrid_payoffs[hybrid_pid]
            seat_stats[hybrid_pid]["h_won"] += h_is_won
            seat_stats[hybrid_pid]["opp_avg_payoff"] += opp_avg_p
            seat_stats[hybrid_pid]["opp_avg_won"] += opp_avg_w

            # 2) Restore the pristine deal and replay with the Pure line-up.
            env.game.restore_checkpoint(init_checkpoint)
            state = env.get_state(env.get_player_id())

            env.set_agents(pure_agents_only)
            pure_payoffs = run_copied_env(env, state)
            if logger:
                logger.log_hvp_game(hybrid_pid, "pure", env.game.players,
                                    pure_payoffs, env.num_players)

            p = env.game.players[hybrid_pid]
            p_is_won = _seat_winner(hybrid_pid, p.bid is not None and p.tricks_won == p.bid)

            pure_stats["payoff"] += pure_payoffs[hybrid_pid]
            pure_stats["won"] += p_is_won
            seat_stats[hybrid_pid]["p_payoff"] += pure_payoffs[hybrid_pid]
            seat_stats[hybrid_pid]["p_won"] += p_is_won

            if (game_idx + 1) % max(1, num_games_per_seat // 5) == 0:
                print(f"  Seat P{hybrid_pid}: Game {game_idx + 1}/{num_games_per_seat} "
                      f"(Paired) complete.")

    # Final averages.
    h_avg_payoff = hybrid_stats["payoff"] / total_games
    h_win_pct = (hybrid_stats["won"] / total_games) * 100
    h_opp_avg_payoff = hybrid_stats["opp_payoff"] / total_games
    h_opp_win_pct = (hybrid_stats["opp_won"] / total_games) * 100

    p_avg_payoff = pure_stats["payoff"] / total_games
    p_win_pct = (pure_stats["won"] / total_games) * 100

    print(f"\n  Arena Results Across All Seats:")
    print(f"    Hybrid MCNFSP : Payoff {h_avg_payoff:.4f} (Won {h_win_pct:.1f}%)")
    print(f"    Pure NFSP     : Payoff {p_avg_payoff:.4f} (Won {p_win_pct:.1f}%)\n")

    per_seat = []
    for sid in range(env.num_players):
        s = seat_stats[sid]
        per_seat.append(
            {
                "h_payoff": s["h_payoff"] / num_games_per_seat,
                "h_win": (s["h_won"] / num_games_per_seat) * 100,
                "opp_payoff": s["opp_avg_payoff"] / num_games_per_seat,
                "opp_win": (s["opp_avg_won"] / num_games_per_seat) * 100,
                "p_payoff": s["p_payoff"] / num_games_per_seat,
                "p_win": (s["p_won"] / num_games_per_seat) * 100,
            }
        )

    return {
        "hybrid": {
            "payoff": h_avg_payoff, "win_pct": h_win_pct,
            "opp_payoff": h_opp_avg_payoff, "opp_win_pct": h_opp_win_pct,
        },
        "pure": {"payoff": p_avg_payoff, "win_pct": p_win_pct},
        "per_seat": per_seat,
    }