/*
description: >
  The news classifier's latest training: how many of Claude's labels it
  learned from, the share of them that were news, the probability at which it
  calls news, the precision and recall of that call and its average precision
  (all on bursts from articles each fold's model didn't train on), and when it
  was trained.
columns:
  samples: {label: Labeled bursts, format: integer}
  news_share: {label: Labeled news, format: percent}
  threshold: {label: Calls news at, format: percent}
  threshold_precision: {label: Precision of its calls, format: percent, better: higher}
  threshold_recall: {label: News it catches, format: percent, better: higher}
  cv_average_precision: {label: Average precision, format: percent, better: higher}
  trained_at: {label: Trained}
*/
SELECT samples, news_share, threshold, threshold_precision, threshold_recall, cv_average_precision, trained_at
FROM gold.news_classifier
WHERE feature = '(intercept)'
