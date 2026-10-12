"""The news classifier (wiki_pulse/classifier.py): trained on Claude's labels, applied to every burst."""

import random
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from agentic_lakehouse.config import load_platform
from agentic_lakehouse.steps import StepContext, run_step
from wiki_pulse.classifier import INTERCEPT, MIN_LABELS, fit, predict

CONFIG = Path(__file__).parents[1] / "config"
NOW = datetime(2026, 10, 12, 12, 0)


def burst(rng: random.Random, label: str) -> dict:
    """A labeled burst: news draws many editors and few reverts; an edit war, the opposite."""
    news = label == "news"
    return {
        "edits_30m": rng.randrange(10, 40) if news else rng.randrange(7, 25),
        "editors_30m": rng.randrange(6, 20) if news else rng.randrange(3, 6),
        "surprise": rng.uniform(5, 25),
        "bot_share": rng.uniform(0, 0.1),
        "anonymous_share": rng.uniform(0, 0.4),
        "revert_share": rng.uniform(0, 0.15) if news else rng.uniform(0.3, 0.7),
        "claude_label": label,
    }


def labeled(n: int, labels=("news", "edit war")) -> pa.Table:
    rng = random.Random(4)
    return pa.Table.from_pylist([burst(rng, labels[i % len(labels)]) for i in range(n)])


def context(rows: pa.Table, messages: list[str] | None = None) -> StepContext:
    step = load_platform(CONFIG).step("news_classifier")
    log = messages.append if messages is not None else (lambda m: None)
    return StepContext(step, NOW, None, query=lambda sql: rows, log=log)


def test_it_learns_what_tells_news_from_an_edit_war():
    model = fit(context(labeled(120)))
    weights = {row["feature"]: row["weight"] for row in model.to_pylist()}
    assert list(weights) == [INTERCEPT, "log_edits", "log_editors", "log_surprise", "bot_share", "anonymous_share", "revert_share"]
    assert weights["revert_share"] < 0 < weights["log_editors"]
    first = model.to_pylist()[0]
    assert first["samples"] == 120 and first["news_share"] == 0.5 and first["training_accuracy"] > 0.9
    rows = model.to_pylist()
    news = {"edits_30m": 30, "editors_30m": 15, "surprise": 12.0, "bot_share": 0.0, "anonymous_share": 0.2, "revert_share": 0.05}
    war = news | {"editors_30m": 3, "revert_share": 0.6}
    assert predict(rows, news) > 0.9 and predict(rows, war) < 0.1


def test_it_waits_for_enough_labels_of_each_kind():
    messages = []
    assert fit(context(labeled(MIN_LABELS - 1), messages)) is None
    assert "29 labeled bursts" in messages[0]
    assert fit(context(labeled(60, labels=("news",)))) is None  # nothing to tell news from


def test_without_a_model_there_is_no_probability():
    assert predict(None, {}) is None


def test_its_weights_pass_the_steps_quality_gates():
    """As `agentic-lakehouse steps` runs it: the declared columns, then quality_rules.yaml."""
    platform = load_platform(CONFIG)
    published = {}

    class Store:
        def last_run(self, table):
            return None

        def publish(self, step, rows, record):
            published[step.output] = rows

        def record(self, step, record):
            raise AssertionError(f"blocked: {record}")

    run = run_step(platform, platform.step("news_classifier"), query=lambda sql: labeled(120), store=Store(), now=NOW, log=lambda m: None)
    assert run.rows == 7 and all(check.passed for check in run.checks)
    assert published["gold.news_classifier"].schema.field("samples").type == pa.int32()
