"""Wikimedia's recent changes into wiki-pulse's edits (wiki_pulse/edits.py)."""

from datetime import UTC, datetime

import pytest

from wiki_pulse.edits import from_recentchange, is_anonymous


def change(**fields) -> dict:
    """A recentchange event as the feed sends it, trimmed to the fields that matter here."""
    return {
        "type": "edit",
        "namespace": 0,
        "title": "Taylor Swift",
        "title_url": "https://en.wikipedia.org/wiki/Taylor_Swift",
        "comment": "/* Career */ added tour dates",
        "timestamp": 1791800000,
        "user": "Example",
        "bot": False,
        "minor": True,
        "length": {"old": 1000, "new": 1250},
        "server_name": "en.wikipedia.org",
        "wiki": "enwiki",
        "meta": {"uri": "https://en.wikipedia.org/wiki/Taylor_Swift", "domain": "en.wikipedia.org"},
    } | fields


def test_an_article_edit_becomes_an_edit():
    assert from_recentchange(change()) == {
        "event_ts": datetime(2026, 10, 12, 10, 13, 20, tzinfo=UTC),
        "wiki": "enwiki",
        "language": "en",
        "title": "Taylor Swift",
        "kind": "edit",
        "editor": "Example",
        "is_bot": False,
        "is_anonymous": False,
        "is_minor": True,
        "is_revert": False,
        "bytes_delta": 250,
        "page_bytes": 1250,
        "edit_summary": "/* Career */ added tour dates",
        "url": "https://en.wikipedia.org/wiki/Taylor_Swift",
    }


def test_a_new_article_counts_its_whole_size_as_added():
    edit = from_recentchange(change(type="new", length={"new": 812}))
    assert (edit["kind"], edit["bytes_delta"], edit["page_bytes"]) == ("new", 812, 812)


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "log"},
        {"type": "categorize"},
        {"namespace": 1},  # a talk page
        {"server_name": "www.wikidata.org", "wiki": "wikidatawiki"},
        {"server_name": "commons.wikimedia.org", "wiki": "commonswiki"},
        {"server_name": "en.wiktionary.org", "wiki": "enwiktionary"},
    ],
)
def test_everything_but_wikipedia_articles_is_left_out(fields):
    assert from_recentchange(change(**fields)) is None


@pytest.mark.parametrize(
    ("summary", "revert"),
    [
        ("Reverted edits by 203.0.113.7 (talk) to last version by Example", True),
        ("Undid revision 1234 by Example (talk)", True),
        ("rv vandalism", True),
        ("Reverse chronological order", False),
        ("", False),
    ],
)
def test_reverts_are_found_by_their_summary(summary, revert):
    assert from_recentchange(change(comment=summary))["is_revert"] is revert


def test_long_summaries_are_cut_and_a_missing_link_falls_back_to_the_title_url():
    edit = from_recentchange(change(comment="x" * 500, meta={}))
    assert len(edit["edit_summary"]) == 300
    assert edit["url"] == "https://en.wikipedia.org/wiki/Taylor_Swift"


@pytest.mark.parametrize(
    ("editor", "anonymous"),
    [("203.0.113.7", True), ("2001:db8::1", True), ("~2026-12345-67", True), ("Example", False), ("ClueBot NG", False), ("", False)],
)
def test_editors_without_an_account(editor, anonymous):
    assert is_anonymous(editor) is anonymous
