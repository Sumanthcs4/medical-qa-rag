import os, sys
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "artifacts/vector_store/faiss_index.idx"
TEXT_CHUNKS_PATH = 'artifacts/vector_store/chunks.pkl'
LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
LLM_MODEL_PATH = 'Sumanth4/Llama-2-7b-Medical-QA-LoRA'
RAW_DATA_PATH = 'data/raw/rag_source'