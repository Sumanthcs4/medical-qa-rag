# src/pipeline/inference_pipeline.py

# --- 1. IMPORTS ---
from src.core.embedding_engine import EmbeddingEngine
from src.core.vector_database import VectorDatabase
from src.core.llm_handler import LlmHandler
from src.core.rag_system import RagSystem
from src.utils.logger import setup_logger

logger = setup_logger()

# --- 2. THE INFERENCEPIPELINE CLASS ---
class InferencePipeline:
    def __init__(self):
        """
        Initializes the entire RAG pipeline by loading all necessary components.
        """
        logger.info("Initializing Inference Pipeline...")

        # Step A: Load the core components (models)
        logger.info("Loading Embedding Engine...")
        self.embedding_engine = EmbeddingEngine()
        
        logger.info("Loading LLM Handler...")
        self.llm_handler = LlmHandler()

        # Step B: Load the knowledge base (Vector DB)
        logger.info("Loading Vector Database...")
        self.vector_database = VectorDatabase()
        index_loaded = self.vector_database.load_index()
        if not index_loaded:
            logger.error("Failed to load the FAISS index. Please run the training pipeline first.")
            raise RuntimeError("Could not load FAISS index.")

        # Step C: Load the orchestrator (RAG System)
        logger.info("Loading RAG System...")
        self.rag_system = RagSystem(
            embedding_engine=self.embedding_engine,
            vector_database=self.vector_database
        )
        
        logger.info("Inference Pipeline Initialized Successfully.")
    def ask_question(self, query: str) -> str:
        logger.info(f"Received new query: '{query}'")
        #Retrieve relevant context using the RAG system
        logger.info("Retrieving relevant context...")
        retrieved_chunks = self.rag_system.retriever(query)
        #Generate answer using the LLM handler
        logger.info("Generating response from LLM...")
        answer = self.llm_handler.generate_response(
            query=query,
            retrieved_chunks=retrieved_chunks
        )
        logger.info("Successfully generated response.")
        return answer
    
    


