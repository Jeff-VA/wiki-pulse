/*
description: >
  Edits, editors and the revert share per day over the last `days`, across
  every Wikipedia or one: today from the real-time tier, earlier days from
  Iceberg, merged into one answer.
params:
  days: {type: int, default: 30, min: 1, max: 365, description: How many days back.}
  wiki: {type: string, default: null, description: "One Wikipedia, e.g. enwiki; every one when empty."}
*/
SELECT date_trunc('day', event_ts) AS day, edit_count, editors, revert_share
FROM edits
WHERE event_ts >= date_trunc('day', now()) - INTERVAL :days DAY
  AND wiki = :wiki
ORDER BY day
