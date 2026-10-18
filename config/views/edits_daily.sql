/*
description: >
  Edits, editors and the revert share for each complete day (UTC) over the
  last `days`, across every Wikipedia or one. Days the data covers only in
  part, today and the day the stack started, are left out, so neither reads as
  a dip.
params:
  days: {type: int, default: 30, min: 1, max: 365, description: How many days back.}
  wiki: {type: string, default: null, description: "One Wikipedia, e.g. enwiki; every one when empty."}
*/
WITH daily AS (
  SELECT date_trunc('day', event_ts) AS day, edit_count, editors, revert_share
  FROM edits
  WHERE event_ts >= date_trunc('day', now()) - INTERVAL :days DAY
    AND event_ts < date_trunc('day', now())
    AND wiki = :wiki
),
covered AS (
  SELECT min(event_ts) AS first_hour FROM gold.wiki_hourly
)
SELECT daily.day, daily.edit_count, daily.editors, daily.revert_share
FROM daily CROSS JOIN covered
WHERE daily.day >= covered.first_hour
ORDER BY daily.day
