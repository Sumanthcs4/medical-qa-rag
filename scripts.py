import os

# Define the folder structure
folders = [
    "data/raw",
    "data/processed",
    "artifacts/fine_tuned_model",
    "artifacts/vector_store",
    "notebooks",
    "src/core",
    "src/pipeline",
    "src/utils",
    "src/exceptions",
    "tests",
    "logs"
]

# Define the files to create (path: file_name)
files = {
    "README.md": "",
    "requirements.txt": "",
    ".env.example": "",
    ".gitignore": "",
    "config.py": "",
    "app.py": "",
    "notebooks/3_model_finetuning.ipynb": "",
    "src/__init__.py": "",
    "src/core/__init__.py": "",
    "src/core/document_processor.py": "",
    "src/core/embedding_engine.py": "",
    "src/core/vector_database.py": "",
    "src/core/llm_handler.py": "",
    "src/core/rag_system.py": "",
    "src/pipeline/__init__.py": "",
    "src/pipeline/training_pipeline.py": "",
    "src/pipeline/inference_pipeline.py": "",
    "src/utils/__init__.py": "",
    "src/utils/text_cleaner.py": "",
    "src/utils/logger.py": "",
    "src/exceptions/__init__.py": "",
    "tests/test_basic.py": "",
    "tests/sample_questions.txt": "",
    "logs/app.log": ""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(content)

print("Project structure created successfully!")
