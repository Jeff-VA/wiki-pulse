/*
description: >
  How well the news classifier's calls match Claude's labels over the last
  `days`, on bursts both judged: precision (of its news calls, the share Claude
  labeled news), recall (of Claude's news, the share it called), the Brier
  score of its probabilities, and how many bursts and news that rests on.
params:
  days: {type: int, default: 7, min: 1, max: 90, description: How far back to look.}
  call_rule: {type: string, default: learned, values: [learned, fixed 0.5], description: "Whose calls to judge: the learned threshold's (the v2 classifier, from October 18, 2026), or the fixed 0.5 before it."}
*/
SELECT evaluated_bursts, evaluated_news, true_positives, news_precision, news_recall, brier_score
FROM classifier_eval
WHERE event_ts >= now() - INTERVAL :days DAY
  AND call_rule = :call_rule
