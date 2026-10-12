/*
description: Bursts over the last `hours` by language, the Wikipedias where articles burst most.
params:
  hours: {type: int, default: 24, min: 1, max: 168, description: How far back to look.}
*/
SELECT language, bursts, news_bursts
FROM trends
WHERE event_ts >= now() - INTERVAL :hours HOUR
ORDER BY bursts DESC
LIMIT 10
