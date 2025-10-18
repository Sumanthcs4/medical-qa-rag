# src/core/document_processor.py

# --- 1. IMPORTS ---
# We keep all the necessary imports from your script.
from pathlib import Path
from langchain.text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from src.utils.logger import setup_logger

logger = setup_logger()

# --- 2. THE CLASS DEFINITION ---
# All logic is now encapsulated within this class.
class DocumentProcessor:
    """
    class to process documents from a given directory.
    It extracts text from PDF files and splits the text into manageable chunks.
    """
    # --- 3. THE CONSTRUCTOR (__init__) ---
    # Its job is to receive the directory path and configure the text splitter.
    def __init__(self, raw_data_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the DocumentProcessor.
        Args:
            raw_data_path (str): The path to the directory containing raw PDF files.
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.raw_data_path = Path(raw_data_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"DocumentProcessor initialized for path: {self.raw_data_path}")

    # --- 4. THE HELPER METHOD (_chunk_text) ---
    # Your chunk_text function is now a "private" method of the class.
    # It's a helper that the main process_documents method will use.
    def _chunk_text(self, text: str) -> list[str]:
        """Splits a single text document into chunks."""
        return self.text_splitter.split_text(text)

    # --- 5. THE MAIN PUBLIC METHOD (process_documents) ---
    # This is the single, powerful method that does all the work in memory.
    def process_documents(self) -> list[str]:
        """
        Processes all PDF files in the raw_data_path directory.
        It extracts text and chunks it, all in memory, without creating intermediate files.
        Returns:
            list[str]: A single list containing all text chunks from all documents.
        """
        all_chunks = []
        logger.info(f"Starting to process PDF files from {self.raw_data_path}...")

        # Your PDF iteration logic is now inside the class method.
        for pdf_path in self.raw_data_path.glob("*.pdf"):
            logger.info(f"Processing file: {pdf_path.name}")
            try:
                # Step A: Extract full text from one PDF.
                doc = fitz.open(pdf_path)
                full_text = "".join(page.get_text() for page in doc)
                doc.close()
                
                if not full_text.strip():
                    logger.warning(f"No text extracted from {pdf_path.name}. Skipping.")
                    continue

                # Step B: Immediately chunk the extracted text.
                chunks = self._chunk_text(full_text)
                all_chunks.extend(chunks)
                logger.info(f"Extracted and chunked {pdf_path.name}, created {len(chunks)} chunks.")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")

        logger.info(f"Completed processing all documents. Total chunks created: {len(all_chunks)}")
        return all_chunks