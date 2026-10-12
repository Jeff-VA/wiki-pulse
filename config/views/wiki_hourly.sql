/*
description: >
  Hourly editing on six of the biggest Wikipedias over the last `hours`, from
  the gold rollup: complete hours that passed their quality checks.
params:
  hours: {type: int, default: 48, min: 6, max: 720, description: How far back to look.}
*/
WITH hourly AS (
  SELECT event_ts AS hour, wiki, edit_count, editors, new_articles,
         CAST(revert_edits AS double) / edit_count AS revert_share,
         CAST(bot_edits AS double) / edit_count AS bot_share,
         CAST(anonymous_edits AS double) / edit_count AS anonymous_share
  FROM gold.wiki_hourly
  WHERE event_ts >= now() - INTERVAL :hours HOUR
    AND wiki IN ('enwiki', 'dewiki', 'frwiki', 'eswiki', 'jawiki', 'ruwiki')
)
SELECT hour, wiki, edit_count, editors, new_articles, revert_share, bot_share, anonymous_share
FROM hourly
ORDER BY hour, wiki
