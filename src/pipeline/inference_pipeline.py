# src/pipeline/inference_pipeline.py
from src.core.embedding_engine import EmbeddingEngine
from src.core.vector_database import VectorDatabase
from src.core.llm_handler import LlmHandler
from src.core.rag_system import RagSystem
from src.utils.logger import setup_logger
from src.pipeline.training_pipeline import run_training_pipeline  # New import

logger = setup_logger()

class InferencePipeline:
    def __init__(self):
        logger.info("Initializing Inference Pipeline...")

        # Load embedding engine and LLM
        logger.info("Loading Embedding Engine...")
        self.embedding_engine = EmbeddingEngine()
        
        logger.info("Loading LLM Handler...")
        self.llm_handler = LlmHandler()

        # Load vector DB
        logger.info("Loading Vector Database...")
        self.vector_database = VectorDatabase()
        if not self.vector_database.load_index():
            logger.warning("FAISS index not found. Running training pipeline...")
            run_training_pipeline()
            if not self.vector_database.load_index():
                logger.error("Failed to create/load FAISS index even after training.")
                raise RuntimeError("Could not load FAISS index.")

        # Load RAG system
        logger.info("Loading RAG System...")
        self.rag_system = RagSystem(
            embedding_engine=self.embedding_engine,
            vector_database=self.vector_database
        )

        logger.info("Inference Pipeline Initialized Successfully.")

    def ask_question(self, query: str) -> str:
        logger.info(f"Received new query: '{query}'")
        # Retrieve relevant context using the RAG system
        logger.info("Retrieving relevant context...")
        retrieved_chunks = self.rag_system.retriever(query)
        # Generate answer using the LLM handler
        logger.info("Generating response from LLM...")
        answer = self.llm_handler.generate_response(
            query=query,
            retrieved_chunks=retrieved_chunks
        )
        logger.info("Successfully generated response.")
        return answer
