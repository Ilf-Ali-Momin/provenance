-- Runs once on first cluster start. ARCHITECTURE.md: pgvector holds the
-- evidence index at 768 dimensions with an HNSW index; the extension has to
-- exist before any migration can reference the type.
CREATE EXTENSION IF NOT EXISTS vector;
