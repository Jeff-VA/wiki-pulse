/*
description: >
  Editing across every Wikipedia in a window measured back from now: the last
  hour by default, or the hour before it (120 and 60) to compare with.
params:
  from_minutes_ago: {type: int, default: 60, min: 1, max: 2880, description: Where the window starts.}
  to_minutes_ago: {type: int, default: 0, min: 0, max: 2880, description: Where it ends; 0 is now.}
*/
SELECT edit_count, editors, articles, new_articles, human_share, revert_share, bot_share, anonymous_share
FROM edits
WHERE event_ts >= now() - INTERVAL :from_minutes_ago MINUTE
  AND event_ts < now() - INTERVAL :to_minutes_ago MINUTE
