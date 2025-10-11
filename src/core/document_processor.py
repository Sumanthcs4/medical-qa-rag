from pathlib import Path
import os, sys
import fitz  # PyMuPDF
import logging
from src.utils.logger import setup_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
logger = setup_logger()
input_dir = Path("D:/Projects/medical-qa-rag/data/raw/rag_source")
output_dir = Path("D:/Projects/medical-qa-rag/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdfs(input_dir, output_dir):
    for pdf_path in input_dir.glob("*.pdf"):
        logger.info(f"Processing {pdf_path.name}")
        
        #opening pdf and extracting text
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        output_path = output_dir / f"{pdf_path.stem}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        logger.info(f"Saved extracted text to {output_path.name}")
        
def chunk_text(full_text: str) -> list[str]:
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(full_text)
    return chunks




if __name__ == "__main__":
    from src.utils.logger import setup_logger

    logger = setup_logger()

    processed_dir = Path("data/processed")
    all_chunks = []

    for file_path in processed_dir.glob("*.txt"):
        logger.info(f"Processing file: {file_path.name}")

        text = file_path.read_text(encoding="utf-8")
        
        chunk_texts = chunk_text(text)  #chunking function

        all_chunks.extend(chunk_texts)
        logger.info(f"Added {len(chunk_texts)} chunks from {file_path.name}")

    logger.info(f" Total chunks collected: {len(all_chunks)}")
    print(f"Total chunks collected: {len(all_chunks)}")
