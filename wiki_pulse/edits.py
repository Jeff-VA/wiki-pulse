"""Wikimedia's recent changes as wiki-pulse's `edits`: one event per edit to a Wikipedia article.

The feed (https://stream.wikimedia.org/v2/stream/recentchange) carries every
change to every Wikimedia wiki. from_recentchange() keeps edits and new pages
in Wikipedia articles (namespace 0) and leaves out the rest: Wikidata, Commons
and the other projects, talk and user pages, log entries and categorizations.
"""

import ipaddress
import re
from datetime import UTC, datetime

# Edit summaries that undo someone else's edit. They're written in many languages; these are the
# English conventions, which other wikis' tools often use too, so the share is a lower bound.
REVERT = re.compile(r"\b(revert(ed|ing|s)?|undid|undo|rv|rvv)\b", re.IGNORECASE)


def from_recentchange(raw: dict) -> dict | None:
    """An article edit from one recentchange event, or None for anything else."""
    if raw.get("type") not in ("edit", "new") or raw.get("namespace") != 0:
        return None
    server = raw.get("server_name") or ""
    if not server.endswith(".wikipedia.org"):
        return None
    length = raw.get("length") or {}
    editor = raw.get("user") or ""
    summary = (raw.get("comment") or "")[:300]
    return {
        "event_ts": datetime.fromtimestamp(raw["timestamp"], UTC),
        "wiki": raw["wiki"],
        "language": server.removesuffix(".wikipedia.org"),
        "title": raw["title"],
        "kind": raw["type"],  # "new" for an article's first edit
        "editor": editor,
        "is_bot": bool(raw.get("bot")),
        "is_anonymous": is_anonymous(editor),
        "is_minor": bool(raw.get("minor")),
        "is_revert": bool(REVERT.search(summary)),
        "bytes_delta": (length.get("new") or 0) - (length.get("old") or 0),
        "page_bytes": length.get("new"),
        "edit_summary": summary,
        "url": (raw.get("meta") or {}).get("uri") or raw.get("title_url"),
    }


def is_anonymous(editor: str) -> bool:
    """Whether an edit came from someone without an account: an IP address, or a temporary
    account (names starting with ~), which Wikipedia uses instead of IPs on some wikis."""
    if editor.startswith("~"):
        return True
    try:
        ipaddress.ip_address(editor)
        return True
    except ValueError:
        return False
