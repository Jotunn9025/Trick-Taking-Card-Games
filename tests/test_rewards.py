"""Dense reward tests — asserted against the CURRENT judger.py values.

Exact-bid scoring table:

============= ================== ========
won, on track (need/more)         +1.0
won, over bid                     -0.5
lost, exactly at bid              +0.5
lost, still need tricks           -0.3
lost, over bid                     0.0
============= ================== ========
"""

from ttcg.game.judger import JudgementJudger
from ttcg.game.player import JudgementPlayer


class TestDenseRewards:
    def test_won_trick_on_track(self):
        """Winning a trick when still needing tricks -> +1.0."""
        player = JudgementPlayer(0)
        player.bid = 3
        player.tricks_won = 2  # remaining_needed = 1 >= 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=True) == 1.0

    def test_won_trick_over_bid(self):
        """Winning a trick when already exceeded the bid -> -0.5."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 3  # remaining_needed = -1 < 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=True) == -0.5

    def test_won_trick_exactly_met(self):
        """Winning the exact trick that meets the bid (remaining = 0) -> +1.0."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 2  # remaining_needed = 0 >= 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=True) == 1.0

    def test_zero_bid_win(self):
        """Bid 0, win a trick -> over bid, -0.5."""
        player = JudgementPlayer(0)
        player.bid = 0
        player.tricks_won = 1  # remaining_needed = -1
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=True) == -0.5

    def test_lost_trick_met_bid(self):
        """Losing a trick when already met the bid -> +0.5 (good to lose)."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 2  # remaining_needed = 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=False) == 0.5

    def test_lost_trick_still_need(self):
        """Losing a trick when still needing tricks -> -0.3."""
        player = JudgementPlayer(0)
        player.bid = 3
        player.tricks_won = 1  # remaining_needed = 2 > 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=False) == -0.3

    def test_lost_trick_already_over(self):
        """Losing a trick when already over the bid -> 0.0 (damage is done)."""
        player = JudgementPlayer(0)
        player.bid = 1
        player.tricks_won = 3  # remaining_needed = -2 < 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=False) == 0.0

    def test_zero_bid_lose(self):
        """Bid 0, lose a trick -> exactly at bid, +0.5."""
        player = JudgementPlayer(0)
        player.bid = 0
        player.tricks_won = 0  # remaining_needed = 0
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=False) == 0.5

    def test_no_bid_no_reward(self):
        """No bid set -> zero reward."""
        player = JudgementPlayer(0)
        player.bid = None
        assert JudgementJudger.compute_dense_trick_reward(player, won_trick=True) == 0.0


class TestRoundScoring:
    def test_round_scoring_exact(self):
        """Round scoring: exact bid -> +1.0 (binary)."""
        players = [JudgementPlayer(0), JudgementPlayer(1)]
        players[0].bid = 3
        players[0].tricks_won = 3
        players[1].bid = 2
        players[1].tricks_won = 2
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0
        assert scores[1] == 1.0

    def test_round_scoring_mixed(self):
        """Round scoring: one exact, one miss."""
        players = [JudgementPlayer(0), JudgementPlayer(1)]
        players[0].bid = 3
        players[0].tricks_won = 3
        players[1].bid = 2
        players[1].tricks_won = 4
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0
        assert scores[1] == -1.0

    def test_bid_heuristic_formula(self):
        """Bid reward matches max(-1.0, 0.5 - 0.3 * |bid - expected_tricks|)."""
        import numpy as np

        from ttcg.game.game import JudgementGame

        game = JudgementGame(num_players=4)
        game.np_random = np.random.RandomState(42)
        game.init_game()

        rnd = game.current_round
        acting = rnd.current_player_id
        hand = game.players[acting].hand
        expected = rnd._estimate_tricks(hand)

        bid_value = 1  # legal for any non-dealer first bidder (num_cards=13)
        game.step(bid_value)
        paid = rnd.dense_rewards[acting]
        expected_reward = max(-1.0, 0.5 - 0.3 * abs(bid_value - expected))
        assert abs(paid - expected_reward) < 1e-6