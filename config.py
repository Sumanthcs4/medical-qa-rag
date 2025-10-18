import os, sys

# Get the absolute path of the current file (inside src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up (to the project root)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Now your paths will be correct
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'rag_source')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "artifacts", "vector_store", "faiss_index.idx")
TEXT_CHUNKS_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'vector_store', 'chunks.pkl')
LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
LLM_MODEL_PATH = 'Sumanth4/Llama-2-7b-Medical-QA-LoRA'
