-- Run at the start of every DuckDB session (DBeaver: the connection's bootstrap queries, one line each).
SET force_download = true;  -- the gateway answers GET, not the HEAD probe DuckDB sends first
ATTACH IF NOT EXISTS 'warehouse' AS lake (TYPE iceberg, ENDPOINT 'http://127.0.0.1:8181', AUTHORIZATION_TYPE 'none');
