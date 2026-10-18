"""classifier_eval (ontology.yaml): the news classifier's quality as governed metrics, checked on
bursts whose answers are known."""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from agentic_lakehouse.compiler import MetricQuery, SqlCompiler
from agentic_lakehouse.config import load_platform
from agentic_lakehouse.schema_compiler import arrow_schema

PLATFORM = load_platform(Path(__file__).parents[1] / "config")
T0 = datetime(2026, 10, 12, 9, 0)

# (the classifier's probability, Claude's label): each outcome, and the bursts left out of judging
BURSTS = [
    (0.9, "news"),  # true positive
    (0.6, "routine"),  # false positive
    (0.2, "news"),  # false negative
    (0.1, "edit war"),  # true negative
    (0.05, "routine"),  # true negative
    (0.3, "unclear"),  # labeled, but with no answer to compare with
    (0.4, None),  # scored, never read by Claude
    (None, "news"),  # labeled before the classifier was first trained
]


def burst(minute: int, p_news: float | None, label: str | None) -> dict:
    return {
        "event_ts": T0 + timedelta(minutes=minute), "wiki": "enwiki", "language": "en",
        "title": f"Article {minute}", "url": f"https://en.wikipedia.org/wiki/Article_{minute}",
        "edits_30m": 12, "editors_30m": 5, "expected_edits": 1.0, "surprise": 8.0,
        "bot_share": 0.0, "anonymous_share": 0.1, "revert_share": 0.0,
        "p_news": p_news, "claude_label": label, "explanation": None,
    }  # fmt: skip


def query(*metrics: str, dimensions: tuple[str, ...] = (), rows: list[dict] | None = None) -> list[dict]:
    rows = rows or [burst(minute, p, label) for minute, (p, label) in enumerate(BURSTS)]
    con = duckdb.connect()
    con.register("t", pa.Table.from_pylist(rows, schema=arrow_schema(PLATFORM.dataset("trends"))))
    question = MetricQuery(metrics=list(metrics), dimensions=list(dimensions), start=T0, end=T0 + timedelta(hours=1))
    sql = SqlCompiler(PLATFORM.ontology).compile(question, engine="duckdb", table="t")
    return con.execute(sql).to_arrow_table().to_pylist()


def test_the_confusion_matrix_counts_only_bursts_both_judged():
    [row] = query(
        "flagged_bursts", "labeled_bursts", "scored_bursts", "evaluated_bursts",
        "true_positives", "false_positives", "false_negatives", "true_negatives", "news_calls", "evaluated_news",
    )  # fmt: skip
    assert row == {
        "flagged_bursts": 8, "labeled_bursts": 7, "scored_bursts": 7, "evaluated_bursts": 5,
        "true_positives": 1, "false_positives": 1, "false_negatives": 1, "true_negatives": 2,
        "news_calls": 2, "evaluated_news": 2,
    }  # fmt: skip


def test_rates_are_over_the_evaluated_bursts():
    [row] = query("news_precision", "news_recall", "news_base_rate", "brier_score", "label_coverage")
    assert row["news_precision"] == pytest.approx(1 / 2)
    assert row["news_recall"] == pytest.approx(1 / 2)
    assert row["news_base_rate"] == pytest.approx(2 / 5)
    # (0.9-1)² + (0.6-0)² + (0.2-1)² + (0.1-0)² + (0.05-0)², over the 5 evaluated
    assert row["brier_score"] == pytest.approx((0.01 + 0.36 + 0.64 + 0.01 + 0.0025) / 5)
    assert row["label_coverage"] == pytest.approx(7 / 8)


def test_the_probability_given_to_each_label():
    [row] = query("p_news_when_news", "p_news_when_other")
    assert row["p_news_when_news"] == pytest.approx((0.9 + 0.2) / 2)
    assert row["p_news_when_other"] == pytest.approx((0.6 + 0.1 + 0.05) / 3)


def test_trends_agreement_is_the_diagonal_of_the_confusion_matrix():
    """The trends model's agreement metric predates classifier_eval; both use the same macros, so
    agreement must be (true positives + true negatives) / evaluated."""
    [row] = query("agreement")
    assert row["agreement"] == pytest.approx((1 + 2) / 5)


def test_score_bands_hold_each_burst_once():
    rows = query("flagged_bursts", dimensions=("score_band",))
    assert {r["score_band"]: r["flagged_bursts"] for r in rows} == {
        "0-10%": 1, "10-30%": 2, "30-50%": 2, "50-70%": 1, "70-100%": 1, "unscored": 1,
    }  # fmt: skip


def test_label_sample_tells_the_random_sample_from_the_first_come_one():
    chances = [None, None, 0.5, 0.5, 0.5, 0.0]  # labeled first come; then a coin flip; then the budget spent
    rows = [burst(i, 0.1, "routine") | {"read_probability": chance} for i, chance in enumerate(chances)]
    result = query("flagged_bursts", dimensions=("label_sample",), rows=rows)
    assert {r["label_sample"]: r["flagged_bursts"] for r in result} == {"first come": 2, "random": 3, "not eligible": 1}


@pytest.mark.parametrize("engine", ["pinot", "trino"])
def test_every_classifier_eval_metric_compiles_for_both_tiers(engine):
    model = next(m for m in PLATFORM.ontology.models if m.name == "classifier_eval")
    dimensions = [d.name for d in model.dimensions if d.name != model.time_dimension]
    question = MetricQuery(metrics=[m.name for m in model.metrics], dimensions=dimensions, start=T0, end=T0 + timedelta(hours=1))
    assert "SELECT" in SqlCompiler(PLATFORM.ontology).compile(question, engine=engine, table="trends").upper()
