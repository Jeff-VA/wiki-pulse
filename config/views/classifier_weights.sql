/*
description: >
  What the news classifier learned: each feature's weight, on a scale where
  every feature has the same spread. Positive pushes a burst toward news,
  negative toward an edit war or routine work.
columns:
  feature: {label: Feature}
  weight: {label: Weight toward news}
*/
SELECT CASE feature
         WHEN 'log_edits' THEN 'Edits (log)'
         WHEN 'log_editors' THEN 'Editors (log)'
         WHEN 'log_surprise' THEN 'Surprise (log)'
         WHEN 'bot_share' THEN 'Bot share'
         WHEN 'anonymous_share' THEN 'Anonymous share'
         WHEN 'revert_share' THEN 'Revert share'
         ELSE feature
       END AS feature,
       weight
FROM gold.news_classifier
WHERE feature <> '(intercept)'
ORDER BY weight DESC
