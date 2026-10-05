"""Hybrid MC-NFSP agent for Judgement (Oh Hell).

Combines Information-Set MCTS with NFSP policy evaluation:

- MCTS explores up to ``max_depth`` of the AGENT'S OWN moves; opponent moves
  inside the tree are selected uniformly among explored/legal children.
  (``_sample_opponent_action`` provides the machinery to sample opponent moves
  from their trained Average Policies; hooking it into the tree is a roadmap
  item — see README.)
- Leaf nodes are evaluated with a blend of DQN Q-values and bid-alignment
  heuristics; terminal states use the official +1/-1 scoring.
- During the bidding phase, ``eval_step`` defers to the expert SL policy to
  avoid the DQN's pessimistic zero-bid bias, and bid actions are rewarded by
  a smooth hand-strength accuracy gradient.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, parent=None, action=None, player_id=None):
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action
        self.player_id: Optional[int] = player_id
        self.children: Dict[int, MCTSNode] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        return self.q_value + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return all(a in self.children for a in legal_actions)


class HybridMCNFSPAgent:
    """Information-Set MCTS guided by a trained NFSP policy evaluator."""

    def __init__(
        self,
        env,
        agent_player_id: int,
        all_nfsp_agents=None,
        num_simulations: int = 200,
        max_depth: int = 100,
        exploration_constant: float = 1.414,
    ):
        self.env = env
        self.agent_player_id = agent_player_id
        self.all_nfsp_agents = all_nfsp_agents
        self.nfsp_agent = all_nfsp_agents[agent_player_id] if all_nfsp_agents else None
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

    def step(self, state) -> int:
        return self._run_mcts(state)

    def eval_step(self, state):
        """
        Split decision-making: Bidding uses the expert SL policy; card play
        uses MCTS + leaf evaluation. Falls back to pure MCTS when no NFSP
        agent is attached (e.g. search-only ablations).
        """
        current_legal = list(state["legal_actions"].keys())
        if (
            current_legal
            and all(a <= 13 for a in current_legal)
            and self.nfsp_agent is not None
        ):
            return self.nfsp_agent.eval_step(state)

        action = self._run_mcts(state)
        return action, {"agent": "hybrid_mc_nfsp"}

    # ------------------------------------------------------------------ search

    def _run_mcts(self, state) -> int:
        legal_actions = list(state["legal_actions"].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = MCTSNode(player_id=self.agent_player_id)

        # Dynamic budget scaling: invest more sims when branching is narrow to
        # cut through determinization noise with fewer legal options.
        n_legal = len(legal_actions)
        if n_legal <= 3:
            scaled_sims = self.num_simulations * 4
        elif n_legal <= 5:
            scaled_sims = self.num_simulations * 2
        else:
            scaled_sims = self.num_simulations

        # One deep copy + checkpoint: N lightweight restores instead of N deep copies.
        game_clone = copy.deepcopy(self.env.game)
        checkpoint = game_clone.save_checkpoint()

        for _ in range(scaled_sims):
            self._determinize(game_clone)
            self._simulate(root, game_clone, legal_actions)
            game_clone.restore_checkpoint(checkpoint)

        if not root.children:
            return int(np.random.choice(legal_actions))

        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _sample_opponent_action(self, game, acting_player_id, legal_actions) -> int:
        """Sample an opponent move from its NFSP Average Policy (or fall back)."""
        if not self.all_nfsp_agents or acting_player_id >= len(self.all_nfsp_agents):
            return int(np.random.choice(legal_actions))

        opp_agent = self.all_nfsp_agents[acting_player_id]
        if opp_agent is None:
            return int(np.random.choice(legal_actions))

        if hasattr(opp_agent, "policy_network"):
            # NFSP agent: sample from the SL network (Average Policy).
            raw_state = game.get_state(acting_player_id)
            extracted = self.env._extract_state(raw_state)
            obs = extracted["obs"]

            obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
            device = next(opp_agent.policy_network.parameters()).device
            obs_tensor = obs_tensor.to(device)

            with torch.no_grad():
                log_probs = opp_agent.policy_network(obs_tensor).cpu().numpy()[0]

            probs = np.exp(log_probs)
            legal_probs = probs[legal_actions]
            sum_probs = legal_probs.sum()
            if sum_probs < 1e-8:
                return int(np.random.choice(legal_actions))

            legal_probs /= sum_probs
            return int(np.random.choice(legal_actions, p=legal_probs))
        elif hasattr(opp_agent, "eval_step"):
            # Generic/rule-based agent logic.
            raw_state = game.get_state(acting_player_id)
            action, _ = opp_agent.eval_step(raw_state)
            if action in legal_actions:
                return action
            return int(np.random.choice(legal_actions))
        else:
            return int(np.random.choice(legal_actions))

    def _determinize(self, game):
        """
        Information-Set MCTS: build the 'true unknown cards' from the full deck
        minus all known cards (our hand, played cards, revealed trump), shuffle,
        and deal back to opponents respecting voids (suits they cannot follow).
        """
        if not game.current_round:
            return

        from ttcg.game.card import JudgementCard

        # 1. Collect all KNOWN cards.
        known_card_ids = set()

        for c in game.players[self.agent_player_id].hand:
            known_card_ids.add(c.card_id)

        if game.current_round.get_trump_card():
            known_card_ids.add(game.current_round.get_trump_card().card_id)

        for trick in game.current_round.trick_history:
            for pid, c in trick:
                known_card_ids.add(c.card_id)

        for pid, c in game.current_round.current_trick:
            known_card_ids.add(c.card_id)

        # 2. Build true unknown cards from the 52-card deck.
        true_unknown_cards = [
            c for c in JudgementCard.get_deck() if c.card_id not in known_card_ids
        ]
        np.random.shuffle(true_unknown_cards)

        # 3. Clear opponent hands and record sizes exactly.
        opp_hand_sizes = {}
        for i, p in enumerate(game.players):
            if i != self.agent_player_id:
                opp_hand_sizes[i] = len(p.hand)
                p.hand = []

        # 4. Deduce voids from trick history (lead suits each opponent skipped).
        voids = {i: set() for i in range(game.num_players)}
        for trick in game.current_round.trick_history:
            if not trick:
                continue
            lead_suit = trick[0][1].suit
            for pid, c in trick:
                if c.suit != lead_suit:
                    voids[pid].add(lead_suit)

        if game.current_round.current_trick:
            lead_suit = game.current_round.current_trick[0][1].suit
            for pid, c in game.current_round.current_trick:
                if c.suit != lead_suit:
                    voids[pid].add(lead_suit)

        # 5. Greedy deal using MRV (most constrained first) to avoid void violations.
        sorted_opps = sorted(opp_hand_sizes.keys(), key=lambda p: len(voids[p]), reverse=True)

        for pid in sorted_opps:
            needed = opp_hand_sizes[pid]
            while needed > 0 and true_unknown_cards:
                valid_idx = next(
                    (i for i, c in enumerate(true_unknown_cards) if c.suit not in voids[pid]),
                    -1,
                )
                if valid_idx != -1:
                    card = true_unknown_cards.pop(valid_idx)
                else:
                    # Fallback: constraint violation if strictly necessary.
                    card = true_unknown_cards.pop(0)
                game.players[pid].hand.append(card)
                needed -= 1

    def _simulate(self, root: MCTSNode, game, legal_actions: List[int]):
        node = root
        depth = 0
        path = [node]

        # ---- Selection ----
        current_legal = legal_actions
        while (
            node.children
            and node.is_fully_expanded(current_legal)
            and not node.is_terminal
        ):
            acting_player = game.get_player_id()
            if acting_player == self.agent_player_id:
                node = node.best_child(self.exploration_constant)
                action = node.action
                depth += 1
            else:
                # ISMCTS: only pick actions legal in the current determinization.
                valid_children_actions = [a for a in node.children if a in current_legal]
                if not valid_children_actions:
                    break
                action = int(np.random.choice(valid_children_actions))
                node = node.children[action]

            game.step(action)
            path.append(node)

            if game.is_over():
                node.is_terminal = True
                break
            current_legal = game._get_legal_actions()

        # ---- Expansion ----
        if not node.is_terminal and not game.is_over() and depth < self.max_depth:
            current_legal = game._get_legal_actions()
            acting_player = game.get_player_id()

            if acting_player == self.agent_player_id:
                unexplored = [a for a in current_legal if a not in node.children]
                action = int(np.random.choice(unexplored)) if unexplored else None
            else:
                # Expand opponents randomly to explore all plausible counter-moves.
                action = int(np.random.choice(current_legal))

            if action is not None and action not in node.children:
                child = MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # ---- Leaf evaluation ----
        # Instead of a full rollout, estimate the leaf's expected return with
        # the NFSP agent's Q-values (blended with the bid-alignment heuristic),
        # which avoids simulating to the end of the round.
        if game.is_over():
            reward = self._score_terminal(game)
        else:
            reward = self._nfsp_evaluate(game)

        reward = float(np.clip(reward, -1.0, 1.0))

        # ---- Backpropagation ----
        for n in path:
            n.visits += 1
            n.total_reward += reward

    # ------------------------------------------------------------------ evaluation

    def _score_terminal(self, game) -> float:
        """Official scoring: +1.0 for exact bid, -1.0 for a miss."""
        from ttcg.game.judger import JudgementJudger

        scores = JudgementJudger.compute_round_scores(game.players)
        return scores[self.agent_player_id]

    def _nfsp_evaluate(self, game) -> float:
        """
        Evaluate a leaf state. At early depths (bidding) rely on a smooth
        'bid accuracy' gradient to avoid the DQN's pessimistic zero-bid bias;
        during play, blend the DQN's strategic Q-value with bid alignment.
        """
        p = game.players[self.agent_player_id]
        hand_power = self._calculate_hand_strength(game)

        if any(pl.bid is None for pl in game.players):
            # Still in the strategic bidding phase.
            if p.bid is None:
                # Evaluating a state before our own bid: use hand power.
                return float(np.clip(hand_power / 13.0, 0.0, 1.0))
            # Evaluating a potential bid just made: reward by accuracy.
            diff = abs(p.bid - hand_power)
            accuracy = 1.0 - (diff / 13.0)
            return float(np.clip(accuracy, 0.0, 1.0))

        # ---- Play phase ----
        q_value = 0.0
        raw_state = game.get_state(self.agent_player_id)
        extracted = self.env._extract_state(raw_state)
        obs = extracted["obs"]
        legal_actions = list(extracted["legal_actions"].keys())

        if legal_actions and hasattr(self.nfsp_agent, "_rl_agent"):
            q_estimator = self.nfsp_agent._rl_agent.q_estimator
            obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
            device = next(q_estimator.qnet.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                q_values = q_estimator.qnet(obs_tensor).cpu().numpy()[0]
            q_value = float(np.max(q_values[legal_actions]))

        # Heuristic bid-alignment (local tracking progress during play).
        alignment = self._heuristic_evaluate(game)

        # Blend strategic Q-value with local trick alignment so the bot does not
        # make 'correct' plays that contradict its bid.
        return 0.7 * q_value + 0.3 * alignment

    # ------------------------------------------------------------------ heuristics

    def _calculate_hand_strength(self, game) -> float:
        """Estimate hand power, weighting high cards and trumps."""
        p = game.players[self.agent_player_id]
        if not p.hand:
            return 0.0

        trump_suit = game.current_round.trump_suit if game.current_round else None

        strength = 0.0
        for card in p.hand:
            rank_idx = card.rank_index  # 2=0, A=12

            if card.suit == trump_suit:
                # Trump bonus: even low trumps are strong.
                strength += 0.7 + 0.3 * (rank_idx / 12.0)
            else:
                if rank_idx >= 11:  # A, K
                    strength += 0.8 * (rank_idx / 12.0)
                elif rank_idx >= 9:  # Q, J
                    strength += 0.3 * (rank_idx / 12.0)
                else:
                    # Garbage cards may still win if others are short.
                    strength += 0.05 * (rank_idx / 12.0)

        return strength

    def _heuristic_evaluate(self, game) -> float:
        """Non-terminal evaluation using bid alignment."""
        p = game.players[self.agent_player_id]

        if p.bid is None:
            return 0.0

        tricks_remaining = 0
        if game.current_round:
            tricks_remaining = game.current_round.num_cards - game.current_round.tricks_played

        needed = p.bid - p.tricks_won

        if needed == 0:
            # Met the bid — reward proportional to how few tricks remain (safer).
            total = game.current_round.num_cards if game.current_round else 1
            safety = 1.0 - (tricks_remaining / max(total, 1))
            return 0.3 + 0.4 * safety
        elif needed > 0:
            if tricks_remaining >= needed:
                achievability = needed / max(tricks_remaining, 1)
                return 0.1 * (1.0 - achievability)
            return -0.5  # impossible to make the bid
        else:
            # Over bid — penalize proportional to how far over.
            return float(np.clip(-0.3 * abs(needed), -1.0, 0.0))