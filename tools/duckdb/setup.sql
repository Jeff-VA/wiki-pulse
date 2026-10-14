-- wiki-pulse in DuckDB: for DBeaver, the DuckDB CLI or a notebook. Run this file once in a DuckDB
-- database, with the stack running (DuckDB CLI: duckdb my.duckdb < setup.sql; DBeaver: open it in the
-- connection's SQL editor, then Execute SQL Script), and session.sql at the start of each session
-- (DBeaver: the connection's bootstrap queries). It gives two ways in:
--
--   lake.<namespace>.<table>        the Iceberg tables, raw.* and gold.*, read directly: history up
--                                   to the last flush or rollup run (minutes behind). Treat them as
--                                   read-only: the flush, rollups and steps own them.
--
--   the gateway at http://127.0.0.1:8000, which adds the live data to history, so views over a model
--   (and SQL that reads one) are live to the second:
--   gateway_view(name, 'p=v&...')   a view's rows, for these parameters ('' for its defaults)
--   lakehouse_sql('SELECT ...')     the rows of SQL written like a view, without saving it as one
--   gateway_views()                 the views you can ask for: what each answers from, its parameters
--
-- A question the gateway can't answer (an unknown view, a bad parameter or query) comes back as one
-- row, in a column named error. Written by `agentic-lakehouse duckdb` from config/deployment.yaml:
-- run that again, rather than editing this file, when the config changes.

INSTALL iceberg;
INSTALL httpfs;
-- Download each answer in one request: the gateway runs the query on every request, so reading
-- an answer in ranges, as DuckDB does by default, would run it several times.
-- Set here too, because creating gateway_views() below reads the gateway's list of views.
SET force_download = true;

-- The S3 login in deployment.yaml, for the files behind lake.*, which DuckDB keeps for every session
CREATE OR REPLACE PERSISTENT SECRET wiki_pulse_s3 (
    TYPE s3,
    KEY_ID 'lakehouse',
    SECRET 'lakehouse-local-secret',
    ENDPOINT '127.0.0.1:9100',
    URL_STYLE 'path',
    USE_SSL false,
    REGION 'us-east-1'
);

-- A view's rows, e.g. SELECT * FROM gateway_view('<view>', 'hours=24')
CREATE OR REPLACE MACRO gateway_view(name, params) AS TABLE
    SELECT * FROM read_parquet('http://127.0.0.1:8000/views/' || name || '?format=parquet&errors=inline&' || params);

-- The rows of SQL written like a view: metrics and dimensions by name from a model, or any SELECT
-- over <namespace>.<table> tables, e.g. SELECT * FROM lakehouse_sql('SELECT ... FROM <model> WHERE ...')
CREATE OR REPLACE MACRO lakehouse_sql(sql) AS TABLE
    SELECT * FROM read_parquet('http://127.0.0.1:8000/sql?format=parquet&errors=inline&q=' || url_encode(sql));

-- The views you can ask for: what each answers from, and its parameters with their defaults. A
-- composed view answers live when one of its parts reads a model (tables are <namespace>.<table>).
-- Last, because creating it reads the gateway's list of views.
CREATE OR REPLACE MACRO gateway_views() AS TABLE
    SELECT v.name,
           CASE WHEN v.source = 'model' THEN 'live + history'
                WHEN v.source = 'composed' AND len([r FOR r IN v.reads IF NOT contains(r, '.')]) > 0
                    THEN 'live + history'
                ELSE 'tables (history)' END AS answers_from,
           array_to_string([p.name || '=' || coalesce(p."default", '') FOR p IN v.params], '&') AS params,
           trim(regexp_replace(v.description, '\s+', ' ', 'g')) AS description
    FROM (SELECT unnest(views) AS v
          FROM read_json('http://127.0.0.1:8000/views',
                         columns = {views: 'STRUCT(name VARCHAR, description VARCHAR, source VARCHAR, reads VARCHAR[], params STRUCT(name VARCHAR, "default" VARCHAR)[])[]'}))
    ORDER BY v.name;
