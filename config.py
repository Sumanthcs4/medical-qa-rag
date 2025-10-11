import os, sys
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "artifacts/vector_store/faiss_index.idx"
TEXT_CHUNKS_PATH = 'artifacts/vector_store/chunks.pkl'
LLM_MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
LLM_MODEL_PATH = 'artifacts/fine_tuned_model/'