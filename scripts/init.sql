-- Initialize pgvector extension and collections table.
-- Called automatically by the postgres Docker container on first start.

CREATE EXTENSION IF NOT EXISTS vector;

-- LangChain PGVector stores its collections in this table.
-- Creating it explicitly ensures the schema is ready before the Python
-- service tries to connect.
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY,
    name TEXT NOT NULL,
    cmetadata JSONB
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    uuid UUID PRIMARY KEY,
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding vector(1536),
    document TEXT,
    cmetadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_embedding_collection
    ON langchain_pg_embedding (collection_id);

-- IVFFlat index for fast approximate nearest-neighbor search on 1536-dim vectors.
-- lists=100 is appropriate for collections up to ~1M rows.
CREATE INDEX IF NOT EXISTS idx_embedding_vector
    ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
