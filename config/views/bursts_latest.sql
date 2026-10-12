/*
description: >
  The latest bursts, newest first: when each was flagged, the article, its
  edits and editors in 30 minutes, how surprising that is, the classifier's
  probability that it's news, and Claude's read. `reading` keeps one kind.
params:
  hours: {type: int, default: 24, min: 1, max: 168, description: How far back to look.}
  reading: {type: string, default: null, values: [news, edit war, routine, unclear, not read], description: "Only bursts Claude read this way; every burst when empty."}
*/
SELECT date_trunc('minute', event_ts) AS flagged_at, url, language, peak_edits, peak_editors,
       peak_surprise, news_probability, reading, explanation
FROM trends
WHERE event_ts >= now() - INTERVAL :hours HOUR
  AND reading = :reading
ORDER BY flagged_at DESC
LIMIT 25
