"""The news classifier (wiki_pulse/classifier.py): trained on Claude's labels, applied to every burst."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from agentic_lakehouse.config import load_platform
from agentic_lakehouse.steps import StepContext, run_step
from wiki_pulse.classifier import (
    FEATURES, INTERCEPT, MIN_LABELS, TARGET_PRECISION, average_precision, choose_threshold, fit, folds, predict,
)  # fmt: skip

CONFIG = Path(__file__).parents[1] / "config"
NOW = datetime(2026, 10, 12, 12, 0)


def burst(rng: random.Random, label: str, article: int) -> dict:
    """A labeled burst: news draws many editors and few reverts, on a large wiki; an edit war, the opposite."""
    news = label == "news"
    return {
        "wiki": "enwiki" if news or rng.random() < 0.4 else "itwiki",
        "title": f"2026 Story {article}" if news and rng.random() < 0.5 else f"Article {article}",
        "edits_30m": rng.randrange(10, 40) if news else rng.randrange(7, 25),
        "editors_30m": rng.randrange(6, 20) if news else rng.randrange(3, 6),
        "expected_edits": rng.uniform(0.5, 1.2),
        "bot_share": rng.uniform(0, 0.1),
        "anonymous_share": rng.uniform(0, 0.4),
        "revert_share": rng.uniform(0, 0.15) if news else rng.uniform(0.3, 0.7),
        "claude_label": label,
    }


def labeled(n: int, labels=("news", "edit war")) -> pa.Table:
    """n labeled bursts, two flags per article, with each wiki's size tier joined in as the step's query does."""
    rng = random.Random(4)
    rows = [burst(rng, labels[(i // 2) % len(labels)], i // 2) for i in range(n)]
    return pa.Table.from_pylist([r | {"size_tier": "large" if r["wiki"] == "enwiki" else "medium"} for r in rows])


def context(rows: pa.Table, messages: list[str] | None = None) -> StepContext:
    step = load_platform(CONFIG).step("news_classifier")
    log = messages.append if messages is not None else (lambda m: None)
    return StepContext(step, NOW, None, query=lambda sql: rows, log=log)


NEWS = {"title": "2026 World Series", "edits_30m": 30, "editors_30m": 15, "expected_edits": 0.7, "bot_share": 0.0,
        "anonymous_share": 0.2, "revert_share": 0.05, "size_tier": "large"}  # fmt: skip
WAR = NEWS | {"title": "Gorham, Maine", "editors_30m": 3, "revert_share": 0.6, "size_tier": "medium"}


def test_it_learns_what_tells_news_from_an_edit_war():
    model = fit(context(labeled(120)))
    weights = {row["feature"]: row["weight"] for row in model.to_pylist()}
    assert list(weights) == [INTERCEPT, "log_edits", "log_editors", "log_expected", "bot_share", "anonymous_share",
                             "revert_share", "large_wiki", "year_in_title"]  # fmt: skip
    assert weights["revert_share"] < 0 < weights["log_editors"] and weights["large_wiki"] > 0
    first = model.to_pylist()[0]
    assert first["samples"] == 120 and first["news_share"] == 0.5 and first["training_accuracy"] > 0.9
    rows = model.to_pylist()
    assert predict(rows, NEWS) > 0.9 and predict(rows, WAR) < 0.1


def test_it_learns_a_threshold_and_judges_itself_on_articles_it_didnt_train_on():
    first = fit(context(labeled(120))).to_pylist()[0]
    assert 0 < first["threshold"] < 1 and first["threshold_precision"] >= TARGET_PRECISION
    assert first["threshold_recall"] > 0.9 and first["cv_average_precision"] > 0.9  # these synthetic labels separate cleanly
    assert fit(context(labeled(120))).to_pylist()[0]["threshold"] == first["threshold"]  # the same labels, the same folds


def test_an_articles_flags_share_a_fold_and_every_fold_gets_news():
    articles = [f"Article {i // 3}" for i in range(60)]
    y = np.array([1.0 if (i // 3) % 4 == 0 else 0.0 for i in range(60)])
    fold = folds(articles, y)
    assert all(len({f for a, f in zip(articles, fold) if a == article}) == 1 for article in set(articles))
    assert {int(f) for f, label in zip(fold, y) if label} == {0, 1, 2, 3, 4}


def test_the_threshold_catches_the_most_news_at_the_target_precision():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # calling news from 0.7: 2 of 3 right, 2 of 3 caught; from 0.4: 3 of 6 right (50%), all 3 caught
    assert choose_threshold(p, y, target=0.5) == (0.4, 0.5, 1.0)
    assert choose_threshold(p, y, target=0.6) == pytest.approx((0.7, 2 / 3, 2 / 3))
    assert average_precision(p, y) == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)  # precision at each news, averaged


def test_it_waits_for_enough_labels_of_each_kind():
    messages = []
    assert fit(context(labeled(MIN_LABELS - 1), messages)) is None
    assert "29 labeled bursts" in messages[0]
    assert fit(context(labeled(60, labels=("news",)))) is None  # nothing to tell news from


def test_without_a_model_there_is_no_probability():
    assert predict(None, {}) is None


def test_a_model_trained_on_features_it_no_longer_computes_scores_nothing():
    """Just after an upgrade, the detector may still hold the old weights until the step retrains."""
    old = [{"feature": INTERCEPT, "weight": 0.1, "center": 0.0, "scale": 1.0},
           {"feature": "log_surprise", "weight": 1.0, "center": 0.0, "scale": 1.0}]  # fmt: skip
    assert "log_surprise" not in FEATURES and predict(old, NEWS) is None


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
    assert run.rows == 1 + len(FEATURES) and all(check.passed for check in run.checks)
    assert published["gold.news_classifier"].schema.field("samples").type == pa.int32()
