/*
description: >
  The news classifier's latest training: how many of Claude's labels it
  learned from, the share of them that were news, how many it then got right,
  and when it was trained.
columns:
  samples: {label: Labeled bursts, format: integer}
  news_share: {label: Labeled news, format: percent}
  training_accuracy: {label: Training accuracy, format: percent, better: higher}
  trained_at: {label: Trained}
*/
SELECT samples, news_share, training_accuracy, trained_at
FROM gold.news_classifier
WHERE feature = '(intercept)'
