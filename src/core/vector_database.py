import os, sys
import faiss
import numpy as np
from src.utils.logger import setup_logger
import pickle
from config import FAISS_INDEX_PATH, TEXT_CHUNKS_PATH
logger = setup_logger() 


class VectorDatabase:
    def __init__(self,index_path=FAISS_INDEX_PATH,chunks_path=TEXT_CHUNKS_PATH):
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
            logger.error(f"Error saving index: {str (e)}")
            
    def load_index(self):
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
                logger.warning("FAISS index or text chunks file does not exist.")
                return
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.text_chunks = pickle.load(f)
            logger.info(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} vectors.")
            logger.info(f"Loaded {len(self.text_chunks)} text chunks from {self.chunks_path}.")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.index = None    