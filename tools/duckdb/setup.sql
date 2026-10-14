-- wiki-pulse in DuckDB, for DBeaver or the DuckDB CLI. Run this file once in a DuckDB database
-- (DBeaver: open it in the connection's SQL editor, then Execute SQL Script). It sets up:
--
--   lake.<namespace>.<table>        the lakehouse's Iceberg tables, raw.* and gold.*, read directly:
--                                   history, as of the last flush (a few minutes behind)
--   gateway_view(name, 'p=v&...')   any view at http://127.0.0.1:8000/views, answered by the gateway:
--                                   views over a model are live to the second, history included
--   gateway_views()                 the views you can ask for: what each reads, its parameters
--
-- Each new session then runs session.sql (DBeaver: the connection's bootstrap queries). The stack
-- must be running. Treat lake.* as read-only: the flush, rollups and steps own those tables.

INSTALL iceberg;
INSTALL httpfs;
SET force_download = true;  -- the gateway answers GET, not the HEAD probe DuckDB sends first

-- The local stack's MinIO login (docker/docker-compose.yml), kept by DuckDB for every session
CREATE OR REPLACE PERSISTENT SECRET wiki_pulse_minio (
    TYPE s3,
    KEY_ID 'lakehouse',
    SECRET 'lakehouse-local-secret',
    ENDPOINT '127.0.0.1:9100',
    URL_STYLE 'path',
    USE_SSL false,
    REGION 'us-east-1'
);

-- A view's rows as a table, e.g. SELECT * FROM gateway_view('wikis_now', 'minutes=1440')
CREATE OR REPLACE MACRO gateway_view(name, params) AS TABLE
    SELECT unnest(rows, recursive := true)
    FROM read_json('http://127.0.0.1:8000/views/' || name || '?' || params);

-- What each view answers, and its parameters with their defaults
CREATE OR REPLACE MACRO gateway_views() AS TABLE
    SELECT v.name,
           CASE v.source WHEN 'model' THEN 'live + history' ELSE 'tables (history)' END AS answers_from,
           array_to_string(list_transform(v.params, p -> p.name || '=' || coalesce(CAST(p."default" AS VARCHAR), '')), '&') AS params,
           v.description
    FROM (SELECT unnest(views) AS v FROM read_json('http://127.0.0.1:8000/views'))
    ORDER BY v.name;
