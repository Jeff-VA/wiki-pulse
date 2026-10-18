/*
description: >
  Whether the news classifier's probabilities mean what they say, over the last
  `days`: in each band of probability, the share of bursts Claude labeled news.
  A calibrated model's 10-30% band holds about 20% news, and the share rises
  band by band.
params:
  days: {type: int, default: 7, min: 1, max: 90, description: How far back to look.}
  call_rule: {type: string, default: learned, values: [learned, fixed 0.5], description: "Whose calls to judge: the learned threshold's (the v2 classifier, from October 18, 2026), or the fixed 0.5 before it."}
*/
SELECT score_band, evaluated_bursts, news_base_rate
FROM classifier_eval
WHERE event_ts >= now() - INTERVAL :days DAY
  AND call_rule = :call_rule
ORDER BY score_band
