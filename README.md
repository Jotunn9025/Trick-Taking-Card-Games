# PANES: Policy-guided Asymmetric Nash Equilibrium Search

> This project was previously titled *Combining Neural Fictitious Self Play with Monte Carlo
> Tree Search for Trick-Taking Card Games*

Reinforcement-learning research codebase for trick-taking card games, combining
**Neural Fictitious Self Play (NFSP)** with **Monte Carlo Tree Search (MCTS)** in a hybrid
inference agent. The environment and methods are developed for **Judgement** (a.k.a. *Oh Hell*),
a four-player game whose forced-bidding and exact-bid scoring make it a hard, largely
uncharted testbed for partial-information RL.

The project is organized as three, mutually reinforcing pieces:

1. **A complete RLCard-compatible Judgement environment** — 454-dim observation space,
   66-dim discrete action space, rules enforced entirely inside the environment, and
   lightweight game-state serialization to support the many rollbacks that tree search needs.
2. **A dense-reward NFSP self-play training pipeline** — dense *bid-alignment* rewards layered
   on top of the sparse binary terminal score to solve the 13-trick credit-assignment problem.
3. **A hybrid MCTS + NFSP inference agent** — Information-Set MCTS with void-aware
   determinization, whose opponent modeling and leaf evaluation are bootstrapped from the
   trained NFSP policies.

The goal of the work is to improve an RL agent's ability to play trick-taking card games by
letting deep self-play provide the learned value/policy signal while tree search supplies the
multi-step lookahead that one-step bootstrapping alone cannot.

> Implementation note: the inference agent corresponds to the `HybridMCNFSPAgent` class
> (`src/ttcg/agents/hybrid.py`). Throughout this document it is referred to simply as the
> **hybrid agent**.

---

## Repository layout

```
.
├── pyproject.toml          # uv project metadata, console scripts, extras
├── uv.lock                 # reproducible dependency lockfile
├── README.md
├── paper/                  # manuscript (compiled PDF lives here)
├── src/ttcg/
│   ├── game/               # RLCard-compatible Judgement environment
│   │   ├── card.py         #   card model (id = suit * 13 + rank)
│   │   ├── dealer.py       #   shuffling / dealing / trump reveal
│   │   ├── player.py       #   hand, bid, tricks_won, score
│   │   ├── judger.py       #   legal actions, trick judging, scoring, dense rewards
│   │   ├── round.py        #   one deal: bidding -> trick-taking + bid heuristic
│   │   ├── game.py         #   orchestrator + checkpoint save/restore
│   │   └── env.py          #   RLCard Env wrapper (454-dim obs) + registration
│   ├── agents/
│   │   ├── replay.py       #   dense-reward trajectory reshaping (dependency-free)
│   │   ├── nfsp.py         #   NFSP training runner, loss tracking, evaluation
│   │   ├── mcts.py         #   pure-MCTS baseline (agent-move depth counting)
│   │   └── hybrid.py       #   hybrid MCTS + NFSP inference agent
│   ├── pipeline/
│   │   ├── train.py        #   Phase 1: NFSP training wrapper
│   │   ├── evaluate.py     #   Phases 2-4: hybrid / pure / paired-arena evaluation
│   │   └── cli.py          #   command-line entry point (`ttcg`)
│   ├── analysis/
│   │   ├── statistics.py   #   McNemar, Clopper-Pearson, bootstrap, loaders
│   │   └── report.py       #   statistical report + plots (`ttcg-analyze`)
│   ├── logging/
│   │   └── game_logger.py  #   per-game CSV logging
│   └── dashboards/         #   optional live monitoring (extra)
│       ├── app.py          #   live training dashboard (`ttcg-dashboard`)
│       └── trends.py       #   trend-slope analysis (`ttcg-trends`)
└── tests/                  # pytest suite (see Testing)
```

---

## Requirements & dependencies

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for environment management
- PyTorch, rlcard (the environment integrates with RLCard's `Env`/`Game` interfaces)

The project is developed and validated entirely in CPU mode. `pip` is a declared dependency
because rlcard's import invokes `pip freeze`.

<details>
<summary>Hardware context</summary>

All training, evaluation, and analysis were developed and run on a single commodity
**16 GB laptop running Linux**, using CPU-only execution (CUDA is disabled). No dedicated GPU is
required to reproduce the pipeline.
</details>

---

## Quickstart

```bash
cd <this repository>
uv sync                      # install core dependencies
uv run pytest                # run the test suite
uv run ttcg --help           # see all pipeline options
```

---

## Usage — training & evaluation

The pipeline runs in four phases. `uv run ttcg ...` drives all of them.

```bash
# Phase 1: train NFSP agents from scratch
uv run ttcg \
    --save-dir checkpoints \
    --nfsp-episodes 5000 \
    --evaluate-every 500 --checkpoint-every 1000 \
    --mcts-simulations 50 --mcts-depth 2 \
    --hybrid-games 10 --eval-games 20 --hybrid-vs-pure-games 25

# Evaluate a saved checkpoint (skips training)
uv run ttcg --save-dir checkpoints --load-checkpoint 600000 \
    --eval-games 20 --hybrid-games 10 --hybrid-vs-pure-games 25

# Resume training with adjusted learning rates
uv run ttcg --save-dir checkpoints --load-checkpoint 600000 --resume-training \
    --nfsp-episodes 200000 \
    --rl-learning-rate 0.001 --sl-learning-rate 0.005 \
    --evaluate-every 500 --checkpoint-every 1000
```

| Phase | Purpose | Artifacts |
| --- | --- | --- |
| 1 | NFSP self-play training (or load / resume a checkpoint) | `checkpoints/`, `training_metrics.csv` |
| 2 | All-hybrid evaluation (every seat uses the hybrid agent) | `logs/eval_games_log.csv` |
| 3 | Pure NFSP baseline (deterministic) | `logs/eval_games_log.csv` |
| 4 | **Paired arena**: hybrid vs pure on identical deals | `logs/hybrid_vs_pure_log.csv` |

Phase 4 is the methodological core: each deal is played twice — once by a single hybrid agent
against three pure-NFSP opponents, and once by four pure-NFSP agents — using game checkpoints
to guarantee the *same card distribution*. This isolates the effect of MCTS lookahead from
positional effects and card luck, and produces the paired outcomes analyzed by
`ttcg-analyze`.

### Analysis

```bash
uv run ttcg-analyze --log-dir ./logs
```

Writes `logs/statistical_summary.txt` and plots into `logs/plots/`, with McNemar's test,
Clopper-Pearson confidence intervals, a bootstrap CI on the win-rate difference, and a
per-seat breakdown.

### Live dashboards (optional)

```bash
uv sync --extra dashboards
uv run ttcg-dashboard --csv checkpoints/training_metrics.csv   # live plotly/Dash app
uv run ttcg-trends    --csv checkpoints/training_metrics.csv   # trend slopes
```

---

## The game: Judgement (Oh Hell)

Judgement is a four-player trick-taking card game for a standard 52-card deck. Each deal,
players first **bid** how many tricks they expect to win, then play tricks; scoring is a binary
contract hit (+1 for an exact bid, −1 otherwise). Overbidding is punished exactly as harshly as
underbidding, so the agent must learn not only when to win tricks but when to *deliberately
lose* them.

- **Setup** — this work fixes the hand size at 13 cards (all 52 dealt). The trump suit rotates
  deterministically Spades → Diamonds → Clubs → Hearts each round.
- **Bidding** (actions 0–13) — sequential; the dealer bids last under the *Hook* rule
  (the sum of all bids must not equal 13), guaranteeing at least one player misses their contract.
- **Playing** (actions 14–65) — the player left of the dealer leads; players must follow suit
  when able, and each trick is won by the highest card of the led suit or the highest trump.
- **Observation** — a 454-dimensional vector in `[0,1]` (hand bitmask, trump indicator, bids,
  tricks won, per-seat current-trick slots, played-card history, round count, phase, player,
  progress, and hand-strength heuristics). See the manuscript for the full decomposition.
- **Action space** — 66 actions with a legal-action mask enforced by the environment.

---

## Method

The approach trains a population of NFSP agents wholly through self-play, then wraps each one
in the hybrid agent for inference.

### NFSP training architecture

Each of the four players runs an independent NFSP agent over a shared multilayer perceptron
(`454 → 1024 → 512 → 256 → 66`). The best-response Q-network is trained via Double-DQN with a
0.995 discount and target refreshes every 500 steps; a supervised average-policy network is
trained on a reservoir buffer to preserve a representative record of past play. Multiplayer
self-play uses anticipatory policy mixing (greedy best response with probability η = 0.15,
average policy otherwise) to keep the population from cycling into exploitable patterns.
Replay/reservoir buffers of 350 000 transitions and batches of 512 are the practical limits for
the reference laptop. Full hyperparameters are tabulated in the manuscript.

### Dense reward shaping

The terminal payoff alone is a sparse, 52-action-horizon signal, so two dense signals are added:

- **Bidding** — a hand-strength heuristic estimates the expected trick count `T̂`; the agent is
  rewarded for bidding close to it (`max(−1.0, 0.5 − 0.3·|b − T̂|)`).
- **Playing** — each resolved trick is scored by whether it moves the running trick count toward
  or away from the bid (rewarding wins while a bid is outstanding and *ducks* once it is met),
  with a heavier penalty for overshooting than for falling short.

Per-trick scores accumulate into a running total, so each transition's reward is the change in
that total, with the binary payoff folded in at the last step. This yields one continuous target
that blends trick-by-trick feedback with the true game outcome.

### Hybrid MCTS + NFSP inference agent

At inference, the trained agent runs **Information-Set MCTS** (ISMCTS) over determinized states:

- **Determinization** — each simulation reconstructs a plausible deal: known cards (the agent's
  hand, exposed trump, played tricks) are removed from the deck, trick-history suit-following
  failures reveal *void* constraints, and the unknown cards are dealt to opponents ordered by how
  constrained they are (MRV), respecting voids without redeals.
- **Asymmetric tree** — only the agent's own actions become nodes; opponent turns are resolved by
  sampling their trained average policies, avoiding a combinatorial four-player tree. Node
  selection uses UCB1, with search capped at a shallow depth (two of the agent's own decisions).
- **Leaf evaluation** — a blend of the learned Q-value and a bid-alignment heuristic,
  `0.7·Q_DQN(s) + 0.3·H_bid(s)`, steering search toward the exact contract; terminal states use
  the official binary score.
- **Dynamic simulation budget** — effort scales up when few legal actions are available (up to 4×
  the base budget when ≤ 3 actions are legal), concentrating search where determinization noise
  is highest.

---

## Results

Quantitative results — arena win rates and bid accuracy, statistical validation (McNemar's test,
confidence intervals), training convergence, bidding-behavior evolution, and the replay
ablation — are reported in the accompanying manuscript, which is the primary reference for this
work:

> **PANES: Policy-guided Asymmetric Nash Equilibrium Search.**
>
> Preprint PDF: `paper/paper.pdf`

The pipeline produces everything needed to reproduce and extend those figures:
`checkpoints/` (models + `training_metrics.csv`), `logs/` (per-game logs), and
`ttcg-analyze` (statistical report + plots).

---

## Testing

```bash
cd <this repository>
uv run pytest          # full suite (requires PyTorch + rlcard)
```

Tests that touch PyTorch/rlcard (`test_env`, `test_hybrid`, NFSP smoke tests) skip cleanly when
those libraries are absent, so the game-logic, statistical, and pure-MCTS suites also run with
just `numpy pandas scipy`:

```bash
uv run pytest tests/test_card.py tests/test_game.py \
    tests/test_rewards.py tests/test_mcts.py tests/test_stats.py tests/test_nfsp.py
```

The suite includes randomized differential tests for trick judging, deal-reproducibility checks,
observation-layout verification, dense-reward trajectory reshaping, and the statistics module.

---

## Project status & roadmap

This repository is the shared foundation for an ongoing line of work and is intended to be the
starting point for further research on RL for trick-taking card games.

Planned directions:

- Extend training to the full multi-deal schedule (the classic varying hand sizes 1 → … → 13 → … → 1)
  behind a configuration flag.
- Improve the ducking/avoidance behavior (the dominant failure mode under exact-bid scoring) via
  hierarchical or adaptively-scaled rewards.
- Move opponent modeling inside the search tree to the learned policies (currently leaf
  evaluation and bid decisions use the trained networks; opponent turns inside the tree are
  selected among legal/explored children).
- Maintain a posterior over hidden opponent cards instead of point-estimate determinization.

---

## Contributing

Bug reports, reproductions, and extensions are welcome. If you build on this work for a
publication, please cite the manuscript above.
