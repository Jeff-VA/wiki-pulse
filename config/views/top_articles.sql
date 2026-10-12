/*
description: >
  The most edited articles of one day across every Wikipedia (`days_ago` 0 is
  today so far, 1 yesterday), with their reverts, their busiest hour, and
  Claude's read if the burst detector flagged them.
params:
  days_ago: {type: int, default: 0, min: 0, max: 6, description: Which day; 0 is today.}
columns:
  edits: {format: integer}
  reverts: {format: integer, better: lower}
  peak_hour_edits: {label: Busiest hour, format: integer}
*/
SELECT rank, url, language, edits, reverts, peak_hour_edits, bursts, reading, explanation
FROM gold.top_articles_daily
WHERE day = date_trunc('day', now()) - INTERVAL :days_ago DAY
ORDER BY rank
LIMIT 25
