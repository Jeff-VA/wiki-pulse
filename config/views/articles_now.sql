/*
description: >
  The most edited articles over the last `minutes`, across every Wikipedia or
  one language's, with how many editors worked on them and how much was reverted.
params:
  minutes: {type: int, default: 60, min: 5, max: 2880, description: How far back to look.}
  language: {type: string, default: null, description: "Only this language's Wikipedia, e.g. en; every one when empty."}
*/
SELECT url, language, edit_count, editors, revert_share
FROM edits
WHERE event_ts >= now() - INTERVAL :minutes MINUTE
  AND language = :language
ORDER BY edit_count DESC
LIMIT 15
