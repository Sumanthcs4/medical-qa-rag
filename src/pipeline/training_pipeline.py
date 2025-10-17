# src/pipeline/train_pipeline.py
from src.utils.logger import setup_logger
from src.core.document_processor import DocumentProcessor
from src.core.embedding_engine import EmbeddingEngine
from src.core.vector_database import VectorDatabase
from config import RAW_DATA_PATH

logger = setup_logger()

def run_training_pipeline():
    logger.info(">>>>>> STAGE 01: Document Processing Started <<<<<<")
    doc_processor = DocumentProcessor(raw_data_path=RAW_DATA_PATH)
    text_chunks = doc_processor.process_documents()
    logger.info(f">>>>>> STAGE 01: Completed. Created {len(text_chunks)} text chunks. <<<<<<")

    logger.info(">>>>>> STAGE 02: Embedding Generation Started <<<<<<")
    embedding_gen = EmbeddingEngine()
    embeddings = embedding_gen.generate_embeddings(text_chunks)
    logger.info(f">>>>>> STAGE 02: Completed. Generated {len(embeddings)} embeddings. <<<<<<")

    logger.info(">>>>>> STAGE 03: Index Building and Saving Started <<<<<<")
    vector_db = VectorDatabase()
    vector_db.build_index(text_chunks, embeddings)
    vector_db.save_index()
    logger.info(">>>>>> STAGE 03: Completed. Index and chunks saved to disk. <<<<<<")
    logger.info("######### TRAINING PIPELINE COMPLETED SUCCESSFULLY #########")
