#'src/core/embedding_engine.py'
import os, sys

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME  # Now works
from src.utils.logger import setup_logger

logger = setup_logger()

logger = setup_logger()

class EmbeddingEngine:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        # Correctly use the imported class directly
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def generate_embedding(self, text_chunks):
        embeddings = self.model.encode(text_chunks)
        logger.info(f"Generated embeddings for {len(text_chunks)} text chunks")
        return embeddings
