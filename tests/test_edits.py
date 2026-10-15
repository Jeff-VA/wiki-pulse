"""Wikimedia's recent changes into wiki-pulse's edits (wiki_pulse/edits.py)."""

from datetime import UTC, datetime

import pytest

from wiki_pulse.edits import KNOWN_FIELDS, from_recentchange, is_anonymous


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


def test_a_field_the_feed_stops_sending_becomes_null_not_a_default():
    edit = from_recentchange({k: v for k, v in change().items() if k not in ("bot", "minor", "length", "user", "comment")})
    for column in ("is_bot", "is_minor", "bytes_delta", "page_bytes", "editor", "is_anonymous", "edit_summary", "is_revert"):
        assert edit[column] is None, column  # so the drift agent sees the column fade


def test_a_field_the_feed_starts_sending_passes_through():
    edit = from_recentchange(change(user_rights="autoconfirmed", patrolled=True, revision={"old": 1, "new": 2}))
    assert edit["user_rights"] == "autoconfirmed"  # new upstream: Iceberg gets a column, the drift agent a signal
    assert "patrolled" not in edit and "revision" not in edit  # known, and left out on purpose


def test_todays_feed_passes_nothing_through():
    # The fields an article edit carried in October 2026, sampled from the live feed.
    full = change(
        **{"$schema": "/mediawiki/recentchange/1.0.0", "id": 1, "parsedcomment": "", "patrolled": True,
           "revision": {"old": 1, "new": 2}, "notify_url": "", "server_url": "", "server_script_path": "/w"}
    )  # fmt: skip
    assert set(full) <= KNOWN_FIELDS
    assert set(from_recentchange(full)) == set(from_recentchange(change()))


@pytest.mark.parametrize(
    ("editor", "anonymous"),
    [("203.0.113.7", True), ("2001:db8::1", True), ("~2026-12345-67", True), ("Example", False), ("ClueBot NG", False), ("", False)],
)
def test_editors_without_an_account(editor, anonymous):
    assert is_anonymous(editor) is anonymous
