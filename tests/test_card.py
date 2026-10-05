"""Tests for JudgementCard."""

from ttcg.game.card import JudgementCard


class TestJudgementCard:
    def test_deck_has_52_cards(self):
        deck = JudgementCard.get_deck()
        assert len(deck) == 52

    def test_unique_card_ids(self):
        deck = JudgementCard.get_deck()
        ids = [c.card_id for c in deck]
        assert len(set(ids)) == 52

    def test_card_id_range(self):
        deck = JudgementCard.get_deck()
        for card in deck:
            assert 0 <= card.card_id < 52

    def test_card_id_mapping(self):
        """card_id = suit_index * 13 + rank_index."""
        card = JudgementCard("H", "A")  # Hearts Ace
        assert card.suit_index == 1  # H is index 1
        assert card.rank_index == 12  # A is index 12
        assert card.card_id == 1 * 13 + 12  # = 25

    def test_card_str(self):
        card = JudgementCard("S", "T")
        assert str(card) == "TS"

    def test_card_equality(self):
        c1 = JudgementCard("D", "7")
        c2 = JudgementCard("D", "7")
        assert c1 == c2

    def test_card_inequality(self):
        c1 = JudgementCard("D", "7")
        c2 = JudgementCard("D", "8")
        assert c1 != c2

    def test_suits_and_ranks(self):
        assert len(JudgementCard.suits) == 4
        assert len(JudgementCard.ranks) == 13