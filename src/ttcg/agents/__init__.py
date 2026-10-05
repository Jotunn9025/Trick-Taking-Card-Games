"""Agent implementations for Judgement (Oh Hell).

Submodules are imported directly (``ttcg.agents.mcts``, ``ttcg.agents.hybrid``,
``ttcg.agents.nfsp``, ``ttcg.agents.replay``); the classes below are exposed
lazily so that importing this package never pulls in torch or rlcard.
"""

__all__ = ["JudgementMCTSAgent", "HybridMCNFSPAgent", "MCTSNode"]

_LAZY = {
    "JudgementMCTSAgent": "mcts",
    "HybridMCNFSPAgent": "hybrid",
    "MCTSNode": "hybrid",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)