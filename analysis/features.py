"""Each burst's features as of the moment the detector flagged it, read from the lakehouse through Trino.

The detector sees the burst's last 30 minutes of edits; the extra features here come from the same
window (bytes added and removed, how spread out the edits are, a revert pattern for more languages),
from the article's editing in the 24 hours before (which the detector would look up in
gold.article_hourly), from earlier flags of the same article, and from the wiki's size tier.
Everything is computed from data before the flag, so nothing leaks from the future, except the size
tier, which is the wiki's last 7 days (tiers barely move).
"""

from pathlib import Path

import pandas as pd
import trino

CACHE = Path(__file__).parent / "out" / "bursts.parquet"

# Revert conventions in edit summaries beyond English: the auto-generated undo and rollback summaries
# of the busiest wikis (de, fr, es, it, pt, pl, nl, ru, uk, ja, zh, he, ar, fa, sv, tr, cs, id, ko, vi).
REVERT_ANY_LANGUAGE = (
    r"(?i)\b(revert(ed|ing|s)?|undid|undo|rv|rvv)\b|rückgängig|zurückgesetzt|revertiert|annul|révoqu|révocation"
    r"|reverti|deshech|deshic|ripristin|desfeit|wycofan|anulowan|ongedaan|teruggedraaid|teruggezet|hersteld tot"
    r"|отмен|откат|скасув|取り消|差し戻|巻き戻|リバート|撤销|撤銷|回退|还原|還原|שוחזר|ביטול|استرجاع|واگردان"
    r"|återställ|geri al|vrácen|membatalkan|되돌|lùi sửa|hủy sửa"
)

SQL = f"""
WITH b AS (
  SELECT event_ts, wiki, language, title, edits_30m, editors_30m, expected_edits, surprise,
         bot_share, anonymous_share, revert_share, p_news, claude_label, read_probability
  FROM raw.trends
),
win AS (
  SELECT b.wiki, b.title, b.event_ts,
         sum(CASE WHEN e.bytes_delta > 0 THEN e.bytes_delta ELSE 0 END) AS bytes_added,
         sum(CASE WHEN e.bytes_delta < 0 THEN -e.bytes_delta ELSE 0 END) AS bytes_removed,
         count_if(e.bytes_delta <= -1000) AS large_removals,
         avg(CASE WHEN e.is_minor THEN 1.0 ELSE 0.0 END) AS minor_share,
         avg(CASE WHEN e.edit_summary IS NULL OR e.edit_summary = '' THEN 1.0 ELSE 0.0 END) AS no_summary_share,
         max(CASE WHEN e.kind = 'new' THEN 1 ELSE 0 END) AS new_page,
         max_by(e.page_bytes, e.event_ts) AS page_bytes,
         date_diff('second', min(e.event_ts), max(e.event_ts)) / 60.0 AS span_minutes,
         avg(CASE WHEN regexp_like(coalesce(e.edit_summary, ''), '{REVERT_ANY_LANGUAGE}') THEN 1.0 ELSE 0.0 END)
           AS revert_share_any_language
  FROM b JOIN raw.edits AS e
    ON e.wiki = b.wiki AND e.title = b.title
   AND e.event_ts > b.event_ts - INTERVAL '30' MINUTE AND e.event_ts <= b.event_ts
  GROUP BY 1, 2, 3
),
prior AS (
  SELECT b.wiki, b.title, b.event_ts, count(e.event_ts) AS prior_24h
  FROM b LEFT JOIN raw.edits AS e
    ON e.wiki = b.wiki AND e.title = b.title
   AND e.event_ts > b.event_ts - INTERVAL '1' DAY AND e.event_ts <= b.event_ts - INTERVAL '30' MINUTE
  GROUP BY 1, 2, 3
),
flags AS (
  SELECT b.wiki, b.title, b.event_ts, count(t.event_ts) AS earlier_flags_24h
  FROM b LEFT JOIN raw.trends AS t
    ON t.wiki = b.wiki AND t.title = b.title
   AND t.event_ts < b.event_ts AND t.event_ts >= b.event_ts - INTERVAL '1' DAY
  GROUP BY 1, 2, 3
)
SELECT b.*, win.bytes_added, win.bytes_removed, win.large_removals, win.minor_share, win.no_summary_share,
       win.new_page, win.page_bytes, win.span_minutes, win.revert_share_any_language,
       prior.prior_24h, flags.earlier_flags_24h, coalesce(s.size_tier, 'unknown') AS size_tier
FROM b
LEFT JOIN win ON win.wiki = b.wiki AND win.title = b.title AND win.event_ts = b.event_ts
LEFT JOIN prior ON prior.wiki = b.wiki AND prior.title = b.title AND prior.event_ts = b.event_ts
LEFT JOIN flags ON flags.wiki = b.wiki AND flags.title = b.title AND flags.event_ts = b.event_ts
LEFT JOIN gold.wiki_sizes AS s ON s.wiki = b.wiki
ORDER BY b.event_ts
"""


def load(refresh: bool = False) -> pd.DataFrame:
    """Every burst with its features, from the cache unless `refresh`."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    connection = trino.dbapi.connect(host="127.0.0.1", port=8080, user="classifier-study", catalog="iceberg")
    cursor = connection.cursor()
    cursor.execute(SQL)
    bursts = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])
    CACHE.parent.mkdir(exist_ok=True)
    bursts.to_parquet(CACHE)
    return bursts


if __name__ == "__main__":
    bursts = load(refresh=True)
    print(f"{len(bursts)} bursts, {bursts['claude_label'].notna().sum()} labeled, written to {CACHE}")
