/*
description: >
  Bursts in a window measured back from now: the last 24 hours by default, or
  the day before (2880 and 1440) to compare with. With how many Claude read,
  the share it labeled news, and how many the classifier called news.
params:
  from_minutes_ago: {type: int, default: 1440, min: 1, max: 10080, description: Where the window starts.}
  to_minutes_ago: {type: int, default: 0, min: 0, max: 10080, description: Where it ends; 0 is now.}
*/
SELECT bursts, burst_articles, read_bursts, news_bursts, news_share, called_news_bursts
FROM trends
WHERE event_ts >= now() - INTERVAL :from_minutes_ago MINUTE
  AND event_ts < now() - INTERVAL :to_minutes_ago MINUTE
