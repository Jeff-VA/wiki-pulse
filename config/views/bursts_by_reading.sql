/*
description: Bursts over the last `hours` by Claude's read of them, with how surprising they were at their peak.
params:
  hours: {type: int, default: 24, min: 1, max: 168, description: How far back to look.}
*/
SELECT reading, bursts, peak_surprise
FROM trends
WHERE event_ts >= now() - INTERVAL :hours HOUR
ORDER BY bursts DESC
