import os, sys 
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from src.utils.logger import setup_logger
logger = setup_logger()
class EmbeddingEngine:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        # Correctly use the imported class directly
        self.model = SentenceTransformer(model_name) # <- Correct
        logger.info(f"Loaded embedding model: {model_name}")

    def generate_embedding(self, text_chunks):
        embeddings = self.model.encode(text_chunks)
        logger.info(f"Generated embeddings for {len(text_chunks)} text chunks")
        return embeddings
        
        
        
        