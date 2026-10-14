-- Run at the start of each DuckDB session, in a database set up with setup.sql (DBeaver: the
-- connection's bootstrap queries, a line each; DuckDB CLI: duckdb -init session.sql my.duckdb).
-- Written by `agentic-lakehouse duckdb` from config/deployment.yaml.
-- Download each answer in one request: the gateway runs the query on every request, so reading
-- an answer in ranges, as DuckDB does by default, would run it several times.
SET force_download = true;
-- lake.<namespace>.<table>: the Iceberg tables, through their REST catalog. Treat them as read-only.
ATTACH IF NOT EXISTS 'warehouse' AS lake (TYPE iceberg, ENDPOINT 'http://127.0.0.1:8181', AUTHORIZATION_TYPE 'none');
