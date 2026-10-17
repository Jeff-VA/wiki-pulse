{#-
  wiki-pulse's SQL macros: business definitions written once and used in
  ontology.yaml, views and steps. A macro here is for every engine;
  pinot.sql, trino.sql or duckdb.sql hold an engine's own version, which wins
  over this one. The comment directly above a macro, with no blank line
  between, describes it to people and agents (GET /catalog, MCP list_metrics).
-#}

{# Who made an edit, broadly: a bot, someone anonymous (an IP address or a temporary account), or a registered editor. #}
{% macro editor_type() %}CASE WHEN is_bot = true THEN 'bot' WHEN is_anonymous = true THEN 'anonymous' ELSE 'registered' END{% endmacro %}

{# An edit by a person with an account: not a bot, not anonymous. #}
{% macro human_edit() %}is_bot = false AND is_anonymous = false{% endmacro %}

{# The site a URL is on, e.g. en.wikipedia.org. #}
{% macro url_host(column) %}regexp_extract({{ column }}, '^https?://([^/]+)', 1){% endmacro %}

{# An edit made from a Wikimedia temporary account: an automatically created account named ~... for editors who aren't logged in, counted as anonymous. Uses the is_temporary flag where producers send it (from around October 2026). For older rows it falls back to the editor name starting with '~'. #}
{% macro temporary_account_edit() %}coalesce(is_temporary, editor LIKE '~%'){% endmacro %}

{# An edit that removes 1,000 bytes or more: blanking, vandalism or its clean-up, or a large trim. About 1% of edits. #}
{% macro large_removal() %}bytes_delta <= -1000{% endmacro %}

{# An edit saved without a summary of what it changed (the feed sends an empty one). #}
{% macro no_summary() %}(edit_summary IS NULL OR edit_summary = ''){% endmacro %}

{# Claude's label for a burst (news, edit war, routine or unclear), or 'not read' when Claude didn't read it. #}
{% macro claude_reading() %}coalesce(claude_label, 'not read'){% endmacro %}

{# The news classifier calls a burst news: a probability of 0.5 or more. The decision threshold, in one place. #}
{% macro called_news() %}p_news >= 0.5{% endmacro %}

{# A burst the classifier can be judged on: it scored it, and Claude labeled it news, edit war or routine (an unclear or unread burst has no answer to compare with). #}
{% macro judged_by_both() %}(p_news IS NOT NULL AND claude_label IN ('news', 'edit war', 'routine')){% endmacro %}

{# 1 for a burst Claude labeled news, 0 otherwise: the outcome the classifier's probability is scored against. #}
{% macro news_outcome() %}CASE WHEN claude_label = 'news' THEN 1.0 ELSE 0.0 END{% endmacro %}
