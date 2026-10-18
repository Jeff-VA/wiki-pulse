/*
description: >
  Breaking now: bursts the news classifier called news in the last `hours`,
  the most confident first, with Claude's read where it has one. The detector
  calls news at the threshold its model learned (catching the most news at 50%
  precision), so about half of these should be news by Claude's judgment.
params:
  hours: {type: int, default: 24, min: 1, max: 168, description: How far back to look.}
*/
SELECT date_trunc('minute', event_ts) AS flagged_at, url, language, news_probability, peak_editors,
       reading, explanation
FROM trends
WHERE event_ts >= now() - INTERVAL :hours HOUR
  AND classifier_says = 'news'
ORDER BY news_probability DESC
LIMIT 25
