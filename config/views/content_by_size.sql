/*
description: >
  What edits change, by wiki size tier, over the last `days` (edit_content):
  how many edits, the median edit size, and the shares marked minor, saved
  without a summary, and removing 1,000 bytes or more.
params:
  days: {type: int, default: 7, min: 1, max: 365, description: How far back to look.}
*/
SELECT size_tier, all_edits, median_edit_size, minor_share, no_summary_share, large_removal_share
FROM edit_content
WHERE event_ts >= now() - INTERVAL :days DAY
ORDER BY size_tier
