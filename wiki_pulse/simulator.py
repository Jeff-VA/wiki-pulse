"""Synthetic edits and alerts, shaped like the real ones: for tests, and for working offline.

ingestion.yaml names these as the datasets' `simulator`s. The live stack never
uses them for `edits` or `trends` (their events come from the feed and the
detector); agentic_lakehouse.testing does, to run every view and dashboard
without the stack.
"""

import random
from datetime import datetime, timedelta

# wiki -> (language, share of edits, sample articles)
WIKIS = {
    "enwiki": ("en", 0.34, ["Taylor Swift", "2026 FIFA World Cup", "Artificial intelligence", "Climate change", "Lionel Messi"]),
    "dewiki": ("de", 0.12, ["Deutschland", "Fußball-Bundesliga", "Berlin", "Bundestagswahl 2025"]),
    "frwiki": ("fr", 0.11, ["Paris", "Tour de France 2026", "Emmanuel Macron", "Ligue 1"]),
    "eswiki": ("es", 0.09, ["España", "Real Madrid Club de Fútbol", "México"]),
    "jawiki": ("ja", 0.08, ["日本", "東京都", "大谷翔平"]),
    "ruwiki": ("ru", 0.08, ["Россия", "Москва"]),
    "itwiki": ("it", 0.07, ["Italia", "Roma"]),
    "zhwiki": ("zh", 0.06, ["中国", "台湾"]),
    "ptwiki": ("pt", 0.05, ["Brasil", "Portugal"]),
}
EDITORS = [f"Editor{i}" for i in range(400)] + [f"203.0.113.{i}" for i in range(60)] + [f"~2026-{i:05d}" for i in range(40)]
BOTS = ["InternetArchiveBot", "Citation bot", "ClueBot NG", "AnomieBOT"]
SUMMARIES = ["copyedit", "Updated results", "added source", "/* Career */ expanded", "Reverted edits by 203.0.113.7", "fix typo", ""]


def edit_session(rng: random.Random, start: datetime) -> list[dict]:
    """One editor's visit to an article: one to four edits. One visit in 400 is a burst instead: a
    story breaking, with dozens of edits by many editors in half an hour."""
    wiki = rng.choices(list(WIKIS), [share for _, share, _ in WIKIS.values()])[0]
    language, _, titles = WIKIS[wiki]
    title = rng.choice(titles) if rng.random() < 0.3 else f"Article {rng.randrange(20000)}"
    if rng.random() < 1 / 400:
        editors = rng.sample(EDITORS, rng.randrange(6, 16))
        count, span = rng.randrange(18, 45), timedelta(minutes=30)
    else:
        editor = rng.choice(BOTS) if rng.random() < 0.12 else rng.choice(EDITORS)
        editors, count, span = [editor], rng.randrange(1, 5), timedelta(minutes=10)
    times = sorted(start + span * rng.random() for _ in range(count))
    return [_edit(rng, wiki, language, title, rng.choice(editors), ts, first=i == 0 and rng.random() < 0.02) for i, ts in enumerate(times)]


def _edit(rng, wiki, language, title, editor, ts, first):
    summary = rng.choice(SUMMARIES)
    delta = rng.randrange(-400, 1600)
    return {
        "event_ts": ts,
        "wiki": wiki,
        "language": language,
        "title": title,
        "kind": "new" if first else "edit",
        "editor": editor,
        "is_bot": editor in BOTS,
        "is_anonymous": editor[0].isdigit() or editor.startswith("~"),
        "is_temporary": editor.startswith("~"),
        "is_minor": rng.random() < 0.3,
        "is_revert": summary.startswith("Reverted"),
        "bytes_delta": delta,
        "page_bytes": rng.randrange(2000, 200000),
        "edit_summary": summary,
        "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }


def alert(rng: random.Random, start: datetime) -> list[dict]:
    """One burst the detector flagged, a few times an hour."""
    if rng.random() > 0.05:
        return []
    wiki = rng.choices(list(WIKIS), [share for _, share, _ in WIKIS.values()])[0]
    language, _, titles = WIKIS[wiki]
    title = rng.choice(titles)
    edits, editors = rng.randrange(8, 45), rng.randrange(3, 16)
    news = rng.random() < 0.5
    return [
        {
            "event_ts": start,
            "wiki": wiki,
            "language": language,
            "title": title,
            "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "edits_30m": edits,
            "editors_30m": editors,
            "expected_edits": 0.8,
            "surprise": round(rng.uniform(4, 25), 2),
            "bot_share": round(rng.uniform(0, 0.2), 3),
            "anonymous_share": round(rng.uniform(0, 0.4), 3),
            "revert_share": round(rng.uniform(0, 0.5 if not news else 0.1), 3),
            "p_news": round(rng.uniform(0.6, 0.95) if news else rng.uniform(0.05, 0.4), 3),
            "claude_label": "news" if news else rng.choice(["edit war", "routine", "unclear"]),
            "explanation": f"Editors are updating {title} as a story develops." if news else f"A dispute over {title}: repeated reverts.",
        }
    ]
