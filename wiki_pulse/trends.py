"""The burst detector: articles getting far more edits than usual, spotted as it happens.

A processor over the edits stream (ingestion.yaml: trends). For every article
it keeps the last 30 minutes of edits. When the count stands far above what's
usual for an article being edited on that wiki at that hour of the day, it
emits a burst: the counts, how surprising they are (in standard deviations),
the news classifier's probability that it's a story breaking, and, for fresh
bursts within an hourly budget, Claude's label and one-line explanation.

What's usual comes from a pipeline step, gold.trend_baselines; the classifier's
weights from another, gold.news_classifier. The detector reloads both every 15
minutes, so the models it applies improve as history and labels accumulate.
"""

import math
from collections import deque
from datetime import datetime, timedelta

from wiki_pulse.classifier import predict
from wiki_pulse.explain import Explainer

WINDOW = timedelta(minutes=30)
MIN_EDITS, MIN_EDITORS = 6, 3  # fewer can't be a burst, however quiet the article usually is
THRESHOLD = 4.0  # standard deviations above the usual count
REPEAT_AFTER = timedelta(hours=2)  # sooner only if a burst grows half as surprising again
RELOAD = timedelta(minutes=15)
FRESH = timedelta(minutes=2)  # older bursts (a replay after a restart) don't spend Claude's budget
DEFAULT = (0.7, 1.2)  # usual 30-minute count for an article being edited (mean, std), until baselines exist


class TrendDetector:
    def __init__(self, context, explainer: Explainer | None = None, use_claude: bool = True):
        self.context = context
        self.explainer = explainer or (Explainer.from_environment(context.log) if use_claude else None)
        self.windows: dict[tuple[str, str], deque] = {}
        self.reported: dict[tuple[str, str], tuple[datetime, float]] = {}
        self.baselines: dict[tuple[str, int], tuple[float, float]] = {}
        self.model: list[dict] | None = None
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
        burst["p_news"] = predict(self.model, burst)
        reading = None
        if self.explainer and self.context.now() - when <= FRESH:
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
            self.model = self.context.query("SELECT feature, weight, center, scale FROM gold.news_classifier").to_pylist() or None
        except Exception:
            self.model = None  # trained once enough bursts have Claude's labels

    def _sweep(self, now: datetime) -> None:
        self.windows = {k: w for k, w in self.windows.items() if w and w[-1]["event_ts"] > now - WINDOW}
        self.reported = {k: r for k, r in self.reported.items() if now - r[0] < REPEAT_AFTER}
        self.swept = now
