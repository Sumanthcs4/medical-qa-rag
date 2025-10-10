from pathlib import Path
import os, sys
import fitz  # PyMuPDF
import logging
from src.utils.logger import setup_logger
logger = setup_logger()
input_dir = Path("D:/Projects/medical-qa-rag/data/raw/rag_source")
output_dir = Path("D:/Projects/medical-qa-rag/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

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