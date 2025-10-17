# src/core/vector_database.py

import os
import faiss
import numpy as np
import pickle
from src.utils.logger import setup_logger
from config import FAISS_INDEX_PATH, TEXT_CHUNKS_PATH

logger = setup_logger()

class VectorDatabase:
    def __init__(self, index_path=FAISS_INDEX_PATH, chunks_path=TEXT_CHUNKS_PATH):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.text_chunks = None
        logger.info("VectorDatabase initialized.")

    def build_index(self, text_chunks, embeddings):
        self.text_chunks = text_chunks
        embedding_array = np.array(embeddings).astype("float32")
        vector_dim = embedding_array.shape[1]
        self.index = faiss.IndexFlatL2(vector_dim)
        self.index.add(embedding_array)
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def save_index(self):
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.text_chunks, f)
            logger.info(f"Saved FAISS index to {self.index_path} and text chunks to {self.chunks_path}.")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def load_index(self):
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
                logger.warning("FAISS index or text chunks file does not exist.")
                return False

            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.text_chunks = pickle.load(f)
            logger.info(f"Loaded FAISS index from {self.index_path}...")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.index = None
            self.text_chunks = None
            return False

    # --- THIS IS THE NEWLY ADDED METHOD ---
    def search(self, query_embedding, k=5):
        """
        Searches the FAISS index for the top k most similar vectors.
        """
        if self.index is None:
            logger.error("Cannot search because the index is not loaded.")
            return []

        try:
            # Convert the query embedding to the 2D NumPy array format FAISS expects
            query_vector = np.array([query_embedding]).astype("float32")

            # Perform the search
            distances, indices = self.index.search(query_vector, k)

            # Retrieve the text chunks using the indices from the search result
            retrieved_chunks = [self.text_chunks[i] for i in indices[0] if i != -1]
            
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return []