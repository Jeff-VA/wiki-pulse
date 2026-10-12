/*
description: Bursts per hour over the last `hours` (complete hours), with how many Claude read and how many it called news.
params:
  hours: {type: int, default: 48, min: 6, max: 168, description: How far back to look.}
*/
SELECT date_trunc('hour', event_ts) AS hour, bursts, read_bursts, news_bursts
FROM trends
WHERE event_ts >= date_trunc('hour', now()) - INTERVAL :hours HOUR
  AND event_ts < date_trunc('hour', now())
ORDER BY hour
