"""Wikimedia's recent changes as wiki-pulse's `edits`: one event per edit to a Wikipedia article.

The feed (https://stream.wikimedia.org/v2/stream/recentchange) carries every
change to every Wikimedia wiki. from_recentchange() keeps edits and new pages
in Wikipedia articles (namespace 0) and leaves out the rest: Wikidata, Commons
and the other projects, talk and user pages, log entries and categorizations.

The engine's drift agent keeps the config in step with what arrives, so this
doesn't hide changes to the feed. A field the feed stops sending becomes null,
not a default, so the agent sees its column fade. A field the feed starts
sending, one not in KNOWN_FIELDS, passes through as it comes, so it reaches
Iceberg as a new column for the agent to find.
"""

import ipaddress
import re
from datetime import UTC, datetime

# Edit summaries that undo someone else's edit. They're written in many languages; these are the
# English conventions, which other wikis' tools often use too, so the share is a lower bound.
REVERT = re.compile(r"\b(revert(ed|ing|s)?|undid|undo|rv|rvv)\b", re.IGNORECASE)

# Every top-level field the feed sent for article edits in October 2026, and the log_* fields of
# log entries. from_recentchange() maps the ones wiki-pulse uses and leaves out the rest on purpose.
KNOWN_FIELDS = frozenset(
    {
        "$schema", "meta", "id", "type", "namespace", "title", "title_url", "comment", "parsedcomment",
        "timestamp", "user", "bot", "minor", "patrolled", "length", "revision", "notify_url", "server_url",
        "server_name", "server_script_path", "wiki", "log_id", "log_type", "log_action", "log_params",
        "log_action_comment",
    }
)  # fmt: skip


def from_recentchange(raw: dict) -> dict | None:
    """An article edit from one recentchange event, or None for anything else."""
    if raw.get("type") not in ("edit", "new") or raw.get("namespace") != 0:
        return None
    server = raw.get("server_name") or ""
    if not server.endswith(".wikipedia.org"):
        return None
    editor = raw.get("user")
    summary = None if raw.get("comment") is None else raw["comment"][:300]
    length = raw.get("length")
    edit = {
        "event_ts": datetime.fromtimestamp(raw["timestamp"], UTC),
        "wiki": raw["wiki"],
        "language": server.removesuffix(".wikipedia.org"),
        "title": raw["title"],
        "kind": raw["type"],  # "new" for an article's first edit
        "editor": editor,
        "is_bot": raw.get("bot"),
        "is_anonymous": None if editor is None else is_anonymous(editor),
        "is_temporary": None if editor is None else editor.startswith("~"),  # a temporary account, also anonymous
        "is_minor": raw.get("minor"),
        "is_revert": None if summary is None else bool(REVERT.search(summary)),
        "bytes_delta": None if length is None else (length.get("new") or 0) - (length.get("old") or 0),
        "page_bytes": None if length is None else length.get("new"),
        "edit_summary": summary,
        "url": (raw.get("meta") or {}).get("uri") or raw.get("title_url"),
    }
    return edit | {key: value for key, value in raw.items() if key not in KNOWN_FIELDS and key not in edit}


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
