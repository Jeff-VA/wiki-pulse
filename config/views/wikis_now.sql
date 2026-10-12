/*
description: The busiest Wikipedias over the last `minutes`, by edits, with their editors, revert share and net bytes added.
params:
  minutes: {type: int, default: 60, min: 5, max: 2880, description: How far back to look.}
*/
SELECT wiki, edit_count, editors, revert_share, net_bytes
FROM edits
WHERE event_ts >= now() - INTERVAL :minutes MINUTE
ORDER BY edit_count DESC
LIMIT 12
