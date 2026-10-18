"""The burst detector: articles getting far more edits than usual, spotted as it happens.

A processor over the edits stream (ingestion.yaml: trends). For every article
it keeps the last 30 minutes of edits. When the count stands far above what's
usual for an article being edited on that wiki at that hour of the day, it
emits a burst: the counts, how surprising they are (in standard deviations),
the news classifier's probability that it's a story breaking and whether that
probability clears the threshold its model learned, and, for fresh
bursts within a budget, Claude's label and one-line explanation. Each burst
records its chance of being read, so the labels are a sample whose bias is
known, and the classifier can be judged without it.

What's usual comes from a pipeline step, gold.trend_baselines; the classifier's
weights and threshold from another, gold.news_classifier; each wiki's size tier,
one of the classifier's features, from gold.wiki_sizes. The detector reloads
them every 15 minutes, so the models it applies improve as history and labels
accumulate.
"""

import math
import random
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta

from wiki_pulse.classifier import predict
from wiki_pulse.explain import Explainer

WINDOW = timedelta(minutes=30)
MIN_EDITS, MIN_EDITORS = 6, 3  # fewer can't be a burst, however quiet the article usually is
THRESHOLD = 4.0  # standard deviations above the usual count
REPEAT_AFTER = timedelta(hours=2)  # sooner only if a burst grows half as surprising again
RELOAD = timedelta(minutes=15)
FRESH = timedelta(minutes=2)  # older bursts (a replay after a restart) don't spend Claude's budget
READ_PROBABILITY = 1.0  # Claude reads each fresh burst with this chance while its budget has room: every one, for now
DEFAULT = (0.7, 1.2)  # usual 30-minute count for an article being edited (mean, std), until baselines exist


class TrendDetector:
    def __init__(self, context, explainer: Explainer | None = None, use_claude: bool = True,
                 chance: Callable[[], float] = random.random, read_probability: float = READ_PROBABILITY):
        self.context = context
        self.explainer = explainer or (Explainer.from_environment(context.log) if use_claude else None)
        self.chance = chance  # a uniform draw in [0, 1), which picks the bursts Claude reads
        self.read_probability = read_probability
        self.windows: dict[tuple[str, str], deque] = {}
        self.reported: dict[tuple[str, str], tuple[datetime, float]] = {}
        self.baselines: dict[tuple[str, int], tuple[float, float]] = {}
        self.model: list[dict] | None = None
        self.threshold: float | None = None  # the probability at which the model calls news
        self.sizes: dict[str, str] = {}  # each wiki's size tier
        self.loaded: datetime | None = None
        self.swept: datetime | None = None

    def process(self, events: list[dict]) -> list[dict]:
        now = events[-1]["event_ts"] if events else self.context.now()
        self._reload(now)
        bursts = []
        for edit in events:
            key = (edit["wiki"], edit["title"])
            window = self.windows.setdefault(key, deque())
            window.append(edit)
            while window[0]["event_ts"] <= edit["event_ts"] - WINDOW:
                window.popleft()
            if burst := self._check(key, window, edit):
                bursts.append(burst)
        if self.swept is None or now - self.swept >= timedelta(minutes=1):
            self._sweep(now)
        return bursts

    def usual(self, wiki: str, hour: int) -> tuple[float, float]:
        """The usual 30-minute count for an article being edited, and its spread."""
        return self.baselines.get((wiki, hour), DEFAULT)

    def _check(self, key, window: deque, edit: dict) -> dict | None:
        edits = len(window)
        editors = {e["editor"] for e in window}
        if edits < MIN_EDITS or len(editors) < MIN_EDITORS:
            return None
        mean, std = self.usual(edit["wiki"], edit["event_ts"].hour)
        surprise = (edits - mean) / std
        when = edit["event_ts"]
        last = self.reported.get(key)
        if surprise < THRESHOLD or (last and when - last[0] < REPEAT_AFTER and surprise < last[1] * 1.5):
            return None
        self.reported[key] = (when, surprise)
        burst = {
            "event_ts": when,
            "wiki": edit["wiki"],
            "language": edit["language"],
            "title": edit["title"],
            "url": edit["url"],
            "edits_30m": edits,
            "editors_30m": len(editors),
            "expected_edits": round(mean, 3),
            "surprise": round(surprise, 2),
            "bot_share": round(sum(e["is_bot"] for e in window) / edits, 3),
            "anonymous_share": round(sum(e["is_anonymous"] for e in window) / edits, 3),
            "revert_share": round(sum(e["is_revert"] for e in window) / edits, 3),
        }
        burst["p_news"] = predict(self.model, burst | {"size_tier": self.sizes.get(edit["wiki"])})
        burst["called_news"] = None if burst["p_news"] is None else burst["p_news"] >= self.threshold
        burst["read_probability"], reading = None, None
        if self.explainer:
            # Reading the first bursts of each hour skewed the labels toward quiet hours. A coin flip, with its odds
            # recorded, makes them a random sample of every burst Claude could have read; 0 where it couldn't.
            eligible = self.context.now() - when <= FRESH and self.explainer.has_budget()
            burst["read_probability"] = self.read_probability if eligible else 0.0
            if eligible and self.chance() < self.read_probability:
                reading = self.explainer.read(burst, [e["edit_summary"] for e in window if e.get("edit_summary")][-12:])
        burst["claude_label"] = reading.category if reading else None
        burst["explanation"] = reading.explanation if reading else None
        return burst

    def _reload(self, now: datetime) -> None:
        if self.loaded and now - self.loaded < RELOAD:
            return
        self.loaded = now
        try:
            rows = self.context.query("SELECT wiki, hour_of_day, mean_edits, std_edits FROM gold.trend_baselines").to_pylist()
            # The baselines count edits per hour; a 30-minute window expects half, with sqrt(1/2) the spread.
            self.baselines = {
                (r["wiki"], int(r["hour_of_day"])): (r["mean_edits"] / 2, max(r["std_edits"] / math.sqrt(2), 0.8)) for r in rows
            }
        except Exception as e:  # the table doesn't exist until the step first runs
            self.context.log(f"trends: no baselines yet, using defaults ({str(e)[:120]})")
        try:
            rows = self.context.query("SELECT feature, weight, center, scale, threshold FROM gold.news_classifier").to_pylist()
            self.model, self.threshold = (rows, rows[0]["threshold"]) if rows and rows[0]["threshold"] is not None else (None, None)
        except Exception:
            self.model, self.threshold = None, None  # trained once enough bursts have Claude's labels
        try:
            self.sizes = {r["wiki"]: r["size_tier"] for r in self.context.query("SELECT wiki, size_tier FROM gold.wiki_sizes").to_pylist()}
        except Exception:
            self.sizes = {}  # a wiki without a tier scores as not large

    def _sweep(self, now: datetime) -> None:
        self.windows = {k: w for k, w in self.windows.items() if w and w[-1]["event_ts"] > now - WINDOW}
        self.reported = {k: r for k, r in self.reported.items() if now - r[0] < REPEAT_AFTER}
        self.swept = now
