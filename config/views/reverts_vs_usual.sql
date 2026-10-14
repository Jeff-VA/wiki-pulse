/*
description: >
  Each Wikipedia's share of edits that undo other edits in the last `hours`,
  against its own last 7 days: which wikis are seeing more reverts than usual
  right now. A composed view: both windows are cut across the tiers by the
  gateway, then compared.
params:
  hours: {type: int, default: 1, min: 1, max: 24, description: The recent window.}
  min_edits: {type: int, default: 100, min: 0, description: Leave out wikis with fewer edits in the recent window.}
columns:
  usual_share: {label: Usual revert share, format: percent, better: lower}
  above_usual: {label: Above usual, format: percent, better: lower}
*/
WITH recent AS (
  SELECT wiki, edit_count, revert_share FROM edits
  WHERE event_ts >= now() - INTERVAL :hours HOUR
),
usual AS (
  SELECT wiki, revert_share AS usual_share FROM edits
  WHERE event_ts >= now() - INTERVAL '7' DAY
)
SELECT recent.wiki, recent.edit_count, recent.revert_share, usual.usual_share,
       recent.revert_share - usual.usual_share AS above_usual
FROM recent JOIN usual ON usual.wiki = recent.wiki
WHERE recent.edit_count >= :min_edits
ORDER BY above_usual DESC
LIMIT 15
