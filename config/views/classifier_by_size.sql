/*
description: >
  The news classifier's precision and recall by wiki size tier over the last
  `days`. Most news labels are on large wikis, so this shows whether the model
  only learned to call news there.
params:
  days: {type: int, default: 7, min: 1, max: 90, description: How far back to look.}
  call_rule: {type: string, default: learned, values: [learned, fixed 0.5], description: "Whose calls to judge: the learned threshold's (the v2 classifier, from October 18, 2026), or the fixed 0.5 before it."}
*/
SELECT size_tier, evaluated_bursts, evaluated_news, news_precision, news_recall
FROM classifier_eval
WHERE event_ts >= now() - INTERVAL :days DAY
  AND call_rule = :call_rule
ORDER BY size_tier
