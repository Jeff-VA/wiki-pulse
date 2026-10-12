"""The burst detector (wiki_pulse/trends.py): articles getting far more edits than usual, as it happens."""

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from agentic_lakehouse.config import load_platform
from agentic_lakehouse.processors import ProcessorContext
from agentic_lakehouse.sources import conform
from wiki_pulse.explain import Reading
from wiki_pulse.trends import DEFAULT, TrendDetector

CONFIG = Path(__file__).parents[1] / "config"
START = datetime(2026, 10, 12, 9, 0)
# What's usual on enwiki at 9:00: 1.5 edits an hour (std 2), so 0.75 in 30 minutes (std 1.41).
# A burst then needs 7 edits: (7 - 0.75) / 1.41 = 4.4 standard deviations.
BASELINES = pa.table({"wiki": ["enwiki"], "hour_of_day": [9], "mean_edits": [1.5], "std_edits": [2.0]})
MODEL = pa.table({
    "feature": ["(intercept)", "log_edits", "log_editors", "log_surprise", "bot_share", "anonymous_share", "revert_share"],
    "weight": [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
    "center": [0.0] * 7,
    "scale": [1.0] * 7,
})


class Claude:
    """Stands in for wiki_pulse.explain.Explainer, recording what it was asked."""

    def __init__(self):
        self.asked = []

    def read(self, burst, summaries):
        self.asked.append((burst["title"], summaries))
        return Reading(category="news", explanation=f"{burst['title']} is in the news.")


def detector(tables=None, explainer=None, now=START + timedelta(minutes=10)):
    tables = {"gold.trend_baselines": BASELINES, "gold.news_classifier": MODEL} if tables is None else tables
    platform = load_platform(CONFIG)
    messages = []

    def query(sql):
        table = next((name for name in tables if name in sql), None)
        if table is None:
            raise RuntimeError(f"Table '{sql.split('FROM ')[-1]}' does not exist")
        return tables[table]

    context = ProcessorContext(platform.dataset("trends"), platform.dataset("edits"), query=query, log=messages.append, now=lambda: now)
    return TrendDetector(context, explainer=explainer, use_claude=False), messages


def edit(minute: float, title="Taylor Swift", editor="Editor1", **fields) -> dict:
    return {
        "event_ts": START + timedelta(minutes=minute),
        "wiki": "enwiki",
        "language": "en",
        "title": title,
        "editor": editor,
        "is_bot": False,
        "is_anonymous": False,
        "is_revert": False,
        "edit_summary": f"edit at {minute}",
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    } | fields


def busy(count: int, start=0.0, editors=4, **fields) -> list[dict]:
    """`count` edits a minute apart by `editors` editors in turn."""
    return [edit(start + i, editor=f"Editor{i % editors}", **fields) for i in range(count)]


def test_a_burst_is_flagged_once_with_its_numbers():
    trends, _ = detector()
    bursts = trends.process(busy(10))
    [burst] = bursts  # at the 7th edit; 8 to 10 aren't half as surprising again
    assert burst["event_ts"] == START + timedelta(minutes=6)
    assert (burst["edits_30m"], burst["editors_30m"], burst["expected_edits"]) == (7, 4, 0.75)
    assert burst["surprise"] == pytest.approx(4.42, abs=0.01)
    assert burst["p_news"] == pytest.approx(0.892, abs=0.001)  # the model: sigmoid(0.5 + 1.0 * log(1 + 4 editors))
    assert (burst["claude_label"], burst["explanation"]) == (None, None)  # no Claude here
    assert conform(load_platform(CONFIG).dataset("trends"), burst) is None  # fits the trends dataset


def test_a_growing_burst_is_flagged_again():
    trends, _ = detector()
    assert len(trends.process(busy(10))) == 1
    [again] = trends.process([edit(10, editor="Editor9")])  # 11 edits: 7.25 standard deviations, 1.5 times 4.42
    assert again["edits_30m"] == 11


@pytest.mark.parametrize(
    "edits",
    [
        busy(20, editors=1),  # one editor saving often isn't a burst
        busy(20, editors=2),
        [edit(i * 15, editor=f"Editor{i}") for i in range(10)],  # ten edits over 2 hours: never 7 in 30 minutes
        [edit(i, title=f"Article {i}", editor=f"Editor{i}") for i in range(30)],  # many articles, one edit each
    ],
)
def test_ordinary_editing_isnt_a_burst(edits):
    trends, _ = detector()
    assert trends.process(edits) == []


def test_without_baselines_or_a_model_it_uses_defaults():
    trends, messages = detector(tables={})
    assert trends.usual("enwiki", 9) == DEFAULT
    [burst] = trends.process(busy(8))  # the default is lower: a burst at the 6th edit, the minimum
    assert burst["expected_edits"] == DEFAULT[0] and burst["p_news"] is None
    assert messages[0].startswith("trends: no baselines yet, using defaults")


def test_claude_reads_fresh_bursts_with_their_summaries():
    claude = Claude()
    trends, _ = detector(explainer=claude)
    [burst] = trends.process(busy(10))
    assert (burst["claude_label"], burst["explanation"]) == ("news", "Taylor Swift is in the news.")
    [(title, summaries)] = claude.asked
    assert title == "Taylor Swift" and summaries[-1] == "edit at 6.0" and len(summaries) == 7


def test_claude_doesnt_spend_its_budget_on_old_bursts():
    claude = Claude()
    trends, _ = detector(explainer=claude, now=START + timedelta(hours=3))  # replaying input from 3 hours ago
    [burst] = trends.process(busy(10))
    assert claude.asked == [] and burst["claude_label"] is None


def test_quiet_seconds_and_old_articles_are_swept_away():
    trends, _ = detector()
    trends.process(busy(3))
    assert trends.windows
    trends.process([edit(45, title="Another article")])  # 45 minutes later: the first article's window is stale
    assert list(trends.windows) == [("enwiki", "Another article")]
    assert trends.process([]) == []  # a quiet second
