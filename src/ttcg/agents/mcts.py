"""Pure MCTS agent for Judgement (Oh Hell).

Key feature: ``max_depth`` counts only the AGENT'S OWN moves, not all moves.
With ``max_depth=2`` the tree explores up to 2 of the agent's own turns ahead;
opponent moves in between do not count toward depth.

Uses full game cloning with lightweight checkpoint restores for proper tree
search, and the same hand-strength bid heuristic as the game's round logic.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional

import numpy as np

from ttcg.game.card import JudgementCard
from ttcg.game.judger import JudgementJudger


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, parent=None, action=None, player_id=None):
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action  # the action that led to this node
        self.player_id: Optional[int] = player_id  # who acted to reach this node
        self.children: Dict[int, MCTSNode] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> "MCTSNode":
        return max(self.children.values(), key=lambda c: c.ucb1(exploration_constant))

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return all(a in self.children for a in legal_actions)


class JudgementMCTSAgent:
    """
    Full MCTS agent that clones the game state for proper tree search.

    ``max_depth`` counts only the agent's own moves. Bid actions are evaluated
    with the hand-strength heuristic from the round logic; card play uses a
    bid-alignment-aware rollout policy.
    """

    def __init__(
        self,
        env,
        agent_player_id,
        num_simulations=200,
        max_depth=2,
        exploration_constant=1.414,
        time_budget=None,
    ):
        self.env = env
        self.agent_player_id = agent_player_id
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.time_budget = time_budget  # optional wall-clock budget (seconds)
        self.use_raw = True

    def step(self, state):
        return self._run_mcts(state)

    def eval_step(self, state):
        action = self._run_mcts(state)
        return action, {"agent": "mcts", "simulations": self.num_simulations}

    # ------------------------------------------------------------------ search

    def _run_mcts(self, state) -> int:
        legal_actions = list(state["legal_actions"].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = MCTSNode()

        # One deep copy + checkpoint: N lightweight restores instead of N deep copies.
        game_clone = copy.deepcopy(self.env.game)
        checkpoint = game_clone.save_checkpoint()

        start_time = time.time()
        sim_count = 0

        if self.time_budget is not None:
            while time.time() - start_time < self.time_budget:
                self._one_simulation(root, game_clone, legal_actions)
                game_clone.restore_checkpoint(checkpoint)
                sim_count += 1
        else:
            for _ in range(self.num_simulations):
                self._one_simulation(root, game_clone, legal_actions)
                game_clone.restore_checkpoint(checkpoint)
                sim_count += 1

        if not root.children:
            return np.random.choice(legal_actions).item()
        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _one_simulation(self, root: MCTSNode, game, legal_actions: List[int]):
        """Run one MCTS simulation with agent-move-only depth counting."""
        node = root
        depth = 0  # counts only OUR moves
        path = [node]

        # --- Selection ---
        current_legal = legal_actions
        while node.children and node.is_fully_expanded(current_legal) and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
            if not game.is_over():
                acting_player = game.get_player_id()
                game.step(node.action)
                if acting_player == self.agent_player_id:
                    depth += 1
            path.append(node)
            if game.is_over():
                node.is_terminal = True
                break
            current_legal = game._get_legal_actions()

        # --- Expansion ---
        if not node.is_terminal and not game.is_over() and depth < self.max_depth:
            current_legal = game._get_legal_actions()
            unexplored = [a for a in current_legal if a not in node.children]
            if unexplored:
                action = np.random.choice(unexplored).item()
                acting_player = game.get_player_id()
                child = MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # --- Rollout (smart heuristic policy with depth limit on our moves) ---
        rollout_depth = depth
        while not game.is_over() and rollout_depth < self.max_depth:
            legal = game._get_legal_actions()
            if not legal:
                break
            acting_player = game.get_player_id()
            action = self._rollout_policy(game, legal, acting_player)
            game.step(action)
            if acting_player == self.agent_player_id:
                rollout_depth += 1

        # --- Evaluate ---
        reward = self._evaluate_state(game)

        # --- Backpropagate ---
        for n in path:
            n.visits += 1
            n.total_reward += reward

    # ------------------------------------------------------------------ policies

    def _rollout_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """Bidding: hand-strength heuristic. Play: bid-alignment-aware selection."""
        if not legal_actions:
            return 0

        if game.current_round and game.current_round.is_bidding:
            return self._rollout_bid_policy(game, legal_actions, acting_player)
        return self._rollout_play_policy(game, legal_actions, acting_player)

    def _rollout_bid_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """Pick the legal bid closest to the hand-strength expected tricks."""
        player = game.players[acting_player]
        trump_suit = game.current_round.trump_suit if game.current_round else None
        expected_tricks = self._estimate_tricks(player.hand, trump_suit)
        return min(legal_actions, key=lambda b: abs(b - expected_tricks))

    def _rollout_play_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """If we need tricks, prefer strong cards; if met/over bid, prefer weak."""
        player = game.players[acting_player]

        if player.bid is None:
            return int(np.random.choice(legal_actions))

        need_tricks = player.bid - player.tricks_won
        trump_suit = game.current_round.trump_suit if game.current_round else None

        scored = []
        for action_id in legal_actions:
            card_id = action_id - JudgementJudger.NUM_BID_ACTIONS
            if card_id < 0 or card_id >= 52:
                scored.append((action_id, 0.0))
                continue

            rank = card_id % 13
            suit_idx = card_id // 13
            strength = rank / 12.0
            is_trump = bool(trump_suit and JudgementCard.suits[suit_idx] == trump_suit)

            if need_tricks > 0:
                score = strength
                if is_trump:
                    score += 0.4
            else:  # met or over bid — dump weak cards
                score = 1.0 - strength
                if is_trump:
                    score -= 0.4
            scored.append((action_id, score))

        # Softmax selection (temperature=0.5 for moderate greediness).
        scores = np.array([s for _, s in scored])
        scores = scores / 0.5
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        idx = np.random.choice(len(scored), p=probs)
        return scored[idx][0]

    def _estimate_tricks(self, hand, trump_suit) -> float:
        """Estimate expected tricks from hand strength (matches round.py heuristic)."""
        expected = 0.0
        for c in hand:
            if c.suit == trump_suit:
                if c.rank_index >= 12:
                    expected += 1.0      # Ace of trump
                elif c.rank_index >= 11:
                    expected += 0.8      # King of trump
                elif c.rank_index >= 9:
                    expected += 0.5      # T, J, Q of trump
                else:
                    expected += 0.2      # low trumps
            else:
                if c.rank_index >= 12:
                    expected += 0.5      # Ace off-suit
                elif c.rank_index >= 10:
                    expected += 0.2      # Q, K off-suit
        return expected

    # ------------------------------------------------------------------ evaluation

    def _evaluate_state(self, game) -> float:
        """Terminal states use official scoring; others the bid-alignment heuristic."""
        if game.is_over():
            return self._score_terminal(game)
        return self._heuristic_evaluate(game)

    def _score_terminal(self, game) -> float:
        """Official scoring: +1.0 for exact bid, -1.0 for a miss."""
        scores = JudgementJudger.compute_round_scores(game.players)
        return scores[self.agent_player_id]

    def _heuristic_evaluate(self, game) -> float:
        """
        Non-terminal evaluation using bid alignment: how well is the agent
        tracking toward its bid?
        """
        p = game.players[self.agent_player_id]

        if p.bid is None:
            return 0.0

        tricks_remaining = 0
        if game.current_round:
            tricks_remaining = game.current_round.num_cards - game.current_round.tricks_played

        needed = p.bid - p.tricks_won

        if needed == 0:
            # Met the bid — reward proportional to how few tricks remain (safer).
            safety = (
                1.0 - (tricks_remaining / max(game.current_round.num_cards, 1))
                if game.current_round
                else 1.0
            )
            return 0.3 + 0.4 * safety
        elif needed > 0:
            if tricks_remaining >= needed:
                achievability = needed / max(tricks_remaining, 1)
                return 0.1 * (1.0 - achievability)
            return -0.5  # impossible to make the bid
        else:
            return -0.3 * min(abs(needed), 3)  # over bid — penalize how much