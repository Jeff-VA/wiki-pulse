/*
description: >
  Bursts by wiki size tier over the last `days`: flags per 10,000 edits (how
  readily a wiki's editing bursts, whatever its size), and from burst_intensity
  the median lift over usual, the share at the detector's 6-edit minimum and
  the share that are revert-heavy.
params:
  days: {type: int, default: 7, min: 1, max: 90, description: How far back to look.}
columns:
  bursts_per_10k_edits: {label: Bursts per 10k edits, format: number}
*/
WITH flags AS (
  SELECT size_tier, burst_flags, median_lift, minimum_size_share, revert_heavy_share
  FROM burst_intensity
  WHERE event_ts >= now() - INTERVAL :days DAY
),
volume AS (
  SELECT size_tier, edit_count
  FROM edits
  WHERE event_ts >= now() - INTERVAL :days DAY
)
SELECT flags.size_tier, flags.burst_flags * 10000.0 / volume.edit_count AS bursts_per_10k_edits,
       flags.burst_flags, flags.median_lift, flags.minimum_size_share, flags.revert_heavy_share
FROM flags JOIN volume USING (size_tier)
ORDER BY flags.size_tier
