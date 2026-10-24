"""Tag parsing and persistence contracts for upload flows."""

from api.upload import _normalise_tags


def test_normalise_tags_accepts_json_array() -> None:
    assert _normalise_tags('["ticas", "Receipts"]') == ["ticas", "Receipts"]


def test_normalise_tags_accepts_comma_separated_text() -> None:
    assert _normalise_tags("ticas, Receipts") == ["ticas", "Receipts"]


def test_normalise_tags_drops_blanks_and_exact_duplicates() -> None:
    assert _normalise_tags([" ticas ", "", "ticas"]) == ["ticas"]
