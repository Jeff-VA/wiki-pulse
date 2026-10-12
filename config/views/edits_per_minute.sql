/*
description: >
  Edits per minute across every Wikipedia over the last `minutes`, by who made
  them: bots, anonymous editors, or editors with accounts. Complete minutes only.
params:
  minutes: {type: int, default: 120, min: 10, max: 1440, description: How far back to look.}
*/
SELECT date_trunc('minute', event_ts) AS minute, editor_type, edit_count
FROM edits
WHERE event_ts >= date_trunc('minute', now()) - INTERVAL :minutes MINUTE
  AND event_ts < date_trunc('minute', now())
ORDER BY minute
