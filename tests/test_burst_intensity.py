"""burst_intensity (ontology.yaml): how big and unusual bursts are, checked on flags whose numbers are known."""

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

# (edits_30m, editors_30m, expected_edits, surprise, bot_share, anonymous_share, revert_share)
FLAGS = [
    (6, 3, 0.75, 5.0, 0.0, 0.0, 0.0),  # at the minimum, mostly registered
    (9, 4, 0.75, 9.0, 0.0, 0.6, 0.5),  # mostly anonymous, revert-heavy
    (6, 3, 0.5, 7.0, 0.3, 0.3, 0.4),  # at the minimum, mixed, revert-heavy
    (20, 12, 1.0, 24.0, 0.0, 0.1, 0.05),  # big and very surprising
    (12, 6, 1.5, 12.0, 0.7, 0.0, 0.0),  # mostly bots
    (7, 3, 0.7, 6.0, 0.0, 0.2, 0.3),  # revert-heavy at exactly 30%
]


def flag(minute: int, edits, editors, expected, surprise, bots, anonymous, reverts) -> dict:
    return {
        "event_ts": T0 + timedelta(minutes=minute), "wiki": "enwiki", "language": "en",
        "title": f"Article {minute}", "url": f"https://en.wikipedia.org/wiki/Article_{minute}",
        "edits_30m": edits, "editors_30m": editors, "expected_edits": expected, "surprise": surprise,
        "bot_share": bots, "anonymous_share": anonymous, "revert_share": reverts,
        "p_news": None, "claude_label": None, "explanation": None,
    }  # fmt: skip


def query(*metrics: str, dimensions: tuple[str, ...] = ()) -> list[dict]:
    rows = [flag(minute, *fields) for minute, fields in enumerate(FLAGS)]
    con = duckdb.connect()
    con.register("t", pa.Table.from_pylist(rows, schema=arrow_schema(PLATFORM.dataset("trends"))))
    question = MetricQuery(metrics=list(metrics), dimensions=list(dimensions), start=T0, end=T0 + timedelta(hours=1))
    sql = SqlCompiler(PLATFORM.ontology).compile(question, engine="duckdb", table="t")
    return con.execute(sql).to_arrow_table().to_pylist()


def test_surprise_and_lift_percentiles():
    """Surprise: 5, 6, 7, 9, 12, 24. Lift (edits / expected): 8, 8, 10, 12, 12, 20. DuckDB interpolates."""
    [row] = query("median_surprise", "p90_surprise", "median_lift")
    assert row["median_surprise"] == pytest.approx((7 + 9) / 2)
    assert row["p90_surprise"] == pytest.approx(12 + 0.5 * (24 - 12))
    assert row["median_lift"] == pytest.approx((10 + 12) / 2)


def test_edits_per_editor_pools_the_flags():
    [row] = query("flag_edits", "flag_editors", "edits_per_editor", "avg_flag_edits")
    assert (row["flag_edits"], row["flag_editors"]) == (60, 31)
    assert row["edits_per_editor"] == pytest.approx(60 / 31)  # not the average of each flag's ratio
    assert row["avg_flag_edits"] == pytest.approx(10)


def test_shares_at_the_minimum_size_and_revert_heavy():
    [row] = query("burst_flags", "minimum_size_flags", "minimum_size_share", "revert_heavy_flags", "revert_heavy_share")
    assert (row["burst_flags"], row["minimum_size_flags"], row["revert_heavy_flags"]) == (6, 2, 3)
    assert (row["minimum_size_share"], row["revert_heavy_share"]) == (pytest.approx(2 / 6), pytest.approx(3 / 6))


@pytest.mark.parametrize(
    ("dimension", "counts"),
    [
        ("surprise_band", {"4-6σ": 1, "6-10σ": 3, "10-20σ": 1, "20σ+": 1}),
        ("size_band", {"6 (minimum)": 2, "7-9": 2, "10-19": 1, "20+": 1}),
        ("mostly_by", {"registered": 3, "anonymous": 1, "mixed": 1, "bots": 1}),
    ],
)
def test_each_flag_falls_in_one_band(dimension, counts):
    assert {r[dimension]: r["burst_flags"] for r in query("burst_flags", dimensions=(dimension,))} == counts


@pytest.mark.parametrize("engine", ["pinot", "trino"])
def test_every_burst_intensity_metric_compiles_for_both_tiers(engine):
    model = next(m for m in PLATFORM.ontology.models if m.name == "burst_intensity")
    dimensions = [d.name for d in model.dimensions if d.name != model.time_dimension]
    question = MetricQuery(metrics=[m.name for m in model.metrics], dimensions=dimensions, start=T0, end=T0 + timedelta(hours=1))
    assert "SELECT" in SqlCompiler(PLATFORM.ontology).compile(question, engine=engine, table="trends").upper()
