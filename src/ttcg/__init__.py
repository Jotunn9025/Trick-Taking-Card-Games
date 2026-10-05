"""ttcg — Reinforcement-learning research codebase for trick-taking card games.

Combines Neural Fictitious Self Play with Monte Carlo Tree Search to improve RL
agents' play in trick-taking card games (Judgement / Oh Hell). Provides:

- ``ttcg.game``    — RLCard-compatible game engine (rules, actions, dense rewards)
- ``ttcg.agents``  — NFSP trainer and the MCTS / hybrid MCTS+NFSP agents
- ``ttcg.pipeline``— 4-phase train & evaluate CLI
- ``ttcg.analysis``— paired-deal statistical analysis (McNemar, CIs, bootstrap)
- ``ttcg.logging`` — per-game CSV logging
- ``ttcg.dashboards`` — optional live monitoring (extra: ``uv sync --extra dashboards``)
"""

__version__ = "0.1.0"