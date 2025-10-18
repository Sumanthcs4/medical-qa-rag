import os, sys
import os, sys
# This finds the root directory of your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Replace the old RAW_DATA_PATH with this one
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'rag_source')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "artifacts/vector_store/faiss_index.idx"
TEXT_CHUNKS_PATH = 'artifacts/vector_store/chunks.pkl'
LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
LLM_MODEL_PATH = 'Sumanth4/Llama-2-7b-Medical-QA-LoRA'


