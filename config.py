import os

# Paths
#PDF_PATH = os.path.join("data", "ai_script.pdf")
VECTOR_DB_DIR = "vector_db"
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200