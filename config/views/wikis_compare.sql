/*
description: >
  Wikipedias compared over the last `hours`, from the gold rollup: the 15 with
  the most edits, and the shares of them by bots, anonymous editors and
  reverts, and the net bytes they added.
params:
  hours: {type: int, default: 24, min: 1, max: 720, description: How far back to look.}
*/
WITH totals AS (
  SELECT wiki, sum(edit_count) AS edit_count, sum(new_articles) AS new_articles,
         CAST(sum(revert_edits) AS double) / sum(edit_count) AS revert_share,
         CAST(sum(bot_edits) AS double) / sum(edit_count) AS bot_share,
         CAST(sum(anonymous_edits) AS double) / sum(edit_count) AS anonymous_share,
         sum(net_bytes) AS net_bytes
  FROM gold.wiki_hourly
  WHERE event_ts >= now() - INTERVAL :hours HOUR
  GROUP BY wiki
)
SELECT wiki, edit_count, new_articles, revert_share, bot_share, anonymous_share, net_bytes
FROM totals
ORDER BY edit_count DESC
LIMIT 15
