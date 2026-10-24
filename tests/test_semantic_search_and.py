"""Unit tests for the boolean-AND query syntax in semantic search.

These tests pin the contract: a query like ``market + nmv`` MUST
prefer paragraphs containing **both** terms over paragraphs that only
contain one of them, regardless of how dense each term is in the
candidate text.
"""
from __future__ import annotations

import pytest

from core.semantic_search import (
    AND_FULL_HIT_BOOST,
    AND_SEPARATOR,
    _and_score_boost,
    parse_and_terms,
)


class TestParseAndTerms:
    def test_empty_query_returns_empty_list(self) -> None:
        assert parse_and_terms("") == []

    def test_no_separator_returns_empty_list(self) -> None:
        assert parse_and_terms("market nmv") == []

    def test_single_separator_returns_two_terms(self) -> None:
        terms = parse_and_terms("market + nmv")
        assert terms == ["market", "nmv"]

    def test_separator_only_token_is_dropped(self) -> None:
        terms = parse_and_terms("market +  + nmv")
        assert terms == ["market", "nmv"]

    def test_terms_preserve_original_case(self) -> None:
        # Original case is kept; matching against documents is done
        # case-insensitively further downstream.
        terms = parse_and_terms("Market + NMV")
        assert terms == ["Market", "NMV"]

    def test_custom_separator(self) -> None:
        terms = parse_and_terms("market & nmv", separator="&")
        assert terms == ["market", "nmv"]

    @pytest.mark.parametrize("sep", [AND_SEPARATOR, "+"])
    def test_default_and_alias_separators_match(self, sep: str) -> None:
        assert parse_and_terms(f"a {sep} b") == ["a", "b"]


class TestAndScoreBoost:
    def test_no_and_terms_yields_zero_boost(self) -> None:
        assert _and_score_boost("market nmv", "any text here") == 0.0

    def test_text_missing_all_terms_yields_zero_boost(self) -> None:
        assert _and_score_boost("market + nmv", "completely unrelated") == 0.0

    def test_text_with_all_terms_gets_full_boost(self) -> None:
        boost = _and_score_boost(
            "market + nmv",
            "market value of the nmv index for the year",
        )
        assert boost >= AND_FULL_HIT_BOOST

    def test_text_with_one_term_smaller_than_full_boost(self) -> None:
        full_hit = _and_score_boost(
            "market + nmv",
            "market and nmv together appear here",
        )
        partial = _and_score_boost(
            "market + nmv",
            "only market appears here",
        )
        assert partial < full_hit
