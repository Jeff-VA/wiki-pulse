"""The project's config, views and dashboards, checked without the Docker stack."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_lakehouse.api_gateway import Project, create_app
from agentic_lakehouse.browser import capture, find_chrome, serving
from agentic_lakehouse.config import load_platform
from agentic_lakehouse.dashboards import dashboard_files
from agentic_lakehouse.testing import standin_gateway

CONFIG = Path(__file__).parents[1] / "config"


@pytest.fixture(scope="module")
def gateway():
    return standin_gateway(load_platform(CONFIG))


def test_the_config_views_and_dashboards_check_out():
    result = Project(CONFIG).check()
    assert result.config_error is None
    assert {d.name: d.problems for d in result.dashboards} == {d.name: [] for d in result.dashboards}


@pytest.fixture(scope="module")
def two_days():
    """Long enough to hold a complete day, at a lower rate to stay quick."""
    return standin_gateway(load_platform(CONFIG), hours=50, rate=0.5)


COMPLETE_DAYS = {"edits_daily"}  # views that leave out days the data covers only in part


def test_every_view_returns_rows(gateway, two_days):
    for name in gateway.views:
        # Daily tables are built from complete hours, so in the day's first hour today has none yet.
        params = {"days_ago": 1} if "days_ago" in gateway.views[name].params and datetime.now(UTC).hour == 0 else {}
        source = two_days if name in COMPLETE_DAYS else gateway
        assert source.view(name, params).rows, f"view {name} returned no rows on simulated events"


@pytest.mark.skipif(find_chrome() is None, reason="needs Chrome or Chromium")
@pytest.mark.parametrize("name", sorted(dashboard_files(CONFIG / "dashboards")))
def test_every_dashboard_draws_without_problems(gateway, name):
    with serving(create_app(gateway, dashboards_dir=CONFIG / "dashboards")) as url:
        shot = capture(f"{url}/d/{name}", timeout=30)
    assert shot.settled and shot.problems == []
