"""edit_content (ontology.yaml): what edits changed, checked on edits whose sizes are known."""

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

# (bytes_delta, page_bytes, is_minor, edit_summary, editor type, kind)
EDITS = [
    (50, 1500, False, "fix", "registered", "edit"),
    (5, 5000, True, "", "registered", "edit"),
    (-1200, 20000, False, "rv vandalism", "registered", "edit"),  # a large removal
    (-30, 60000, True, "typo", "bot", "edit"),
    (0, 8000, False, "", "anonymous", "edit"),
    (15000, 15000, False, "new article", "registered", "new"),
    (-2500, 3000, False, "", "anonymous", "edit"),  # a large removal
    (300, 700, False, "expand", "registered", "edit"),
]


def edit(minute: int, delta: int, page: int, minor: bool, summary: str, editor: str, kind: str) -> dict:
    return {
        "event_ts": T0 + timedelta(minutes=minute), "wiki": "enwiki", "language": "en",
        "title": f"Article {minute}", "kind": kind, "editor": f"{editor} {minute}",
        "is_bot": editor == "bot", "is_anonymous": editor == "anonymous", "is_minor": minor,
        "is_revert": summary.startswith("rv"), "bytes_delta": delta, "page_bytes": page,
        "edit_summary": summary, "url": f"https://en.wikipedia.org/wiki/Article_{minute}",
        "is_temporary": editor == "anonymous",
    }  # fmt: skip


def query(*metrics: str, dimensions: tuple[str, ...] = ()) -> list[dict]:
    rows = [edit(minute, *fields) for minute, fields in enumerate(EDITS)]
    con = duckdb.connect()
    con.register("t", pa.Table.from_pylist(rows, schema=arrow_schema(PLATFORM.dataset("edits"))))
    question = MetricQuery(metrics=list(metrics), dimensions=list(dimensions), start=T0, end=T0 + timedelta(hours=1))
    sql = SqlCompiler(PLATFORM.ontology).compile(question, engine="duckdb", table="t")
    return con.execute(sql).to_arrow_table().to_pylist()


def test_bytes_added_and_removed_are_kept_apart():
    [row] = query("all_edits", "bytes_added", "bytes_removed")
    assert (row["all_edits"], row["bytes_added"], row["bytes_removed"]) == (8, 50 + 5 + 15000 + 300, 1200 + 30 + 2500)
    [edits] = query("net_bytes")  # the edits model's net over the same edits (a query reads one model)
    assert edits["net_bytes"] == row["bytes_added"] - row["bytes_removed"]


def test_shares_of_minor_unsummarized_and_large_removing_edits():
    [row] = query("minor_share", "no_summary_share", "large_removal_share")
    assert row == {"minor_share": pytest.approx(2 / 8), "no_summary_share": pytest.approx(3 / 8), "large_removal_share": pytest.approx(2 / 8)}


def test_edit_size_percentiles_are_of_bytes_changed_either_way():
    """Sizes changed: 0, 5, 30, 50, 300, 1200, 2500, 15000. DuckDB interpolates (quantile_cont)."""
    [row] = query("median_edit_size", "p90_edit_size")
    assert row["median_edit_size"] == pytest.approx((50 + 300) / 2)
    assert row["p90_edit_size"] == pytest.approx(2500 + 0.3 * (15000 - 2500))


@pytest.mark.parametrize(
    ("dimension", "counts"),
    [
        ("edit_size", {"none": 1, "1-9 B": 1, "10-99 B": 2, "100-999 B": 1, "1-10 kB": 2, "10 kB+": 1}),
        ("article_size", {"stub (<2 kB)": 2, "short (2-10 kB)": 3, "medium (10-50 kB)": 2, "long (50 kB+)": 1}),
        ("change", {"added": 4, "removed": 3, "no change": 1}),
        ("marked_minor", {"minor": 2, "not minor": 6}),
    ],
)
def test_each_edit_falls_in_one_band(dimension, counts):
    assert {r[dimension]: r["all_edits"] for r in query("all_edits", dimensions=(dimension,))} == counts


def test_large_removals_by_who_made_them():
    rows = query("large_removal_share", dimensions=("editor_type",))
    assert {r["editor_type"]: r["large_removal_share"] for r in rows} == {
        "anonymous": pytest.approx(1 / 2), "registered": pytest.approx(1 / 5), "bot": 0,
    }  # fmt: skip


@pytest.mark.parametrize("engine", ["pinot", "trino"])
def test_every_edit_content_metric_compiles_for_both_tiers(engine):
    model = next(m for m in PLATFORM.ontology.models if m.name == "edit_content")
    dimensions = [d.name for d in model.dimensions if d.name != model.time_dimension]
    question = MetricQuery(metrics=[m.name for m in model.metrics], dimensions=dimensions, start=T0, end=T0 + timedelta(hours=1))
    assert "SELECT" in SqlCompiler(PLATFORM.ontology).compile(question, engine=engine, table="edits").upper()
