import os, sys 
from transformers import sentence_transformers
from config import EMBEDDING_MODEL_NAME
from utils.logger import setup_logger
logger = setup_logger()
class EmbeddingEngine:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = sentence_transformers.SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    def generate_embedding(self, text_chunks):
        return self.model.encode(text_chunks)
        logger.info(f"Generated embeddings for {len(text_chunks)} text chunks")
        
        
        