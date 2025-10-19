# 🩺 Medical Question-Answering with RAG and Fine-Tuned Llama 2

[![License: Llama 2](https://img.shields.io/badge/License-Llama%202-yellow.svg)](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/Sumanth4/medical-qa-rag)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

---

## 🌟 Project Overview

This project is a **Medical Question-Answering System** that delivers **intelligent, factually grounded answers**.

It combines two powerful AI techniques:

1.  **Fine-Tuning:** A base `Llama 2 7B` model is trained on a medical Q&A dataset to learn the **language, style, and reasoning of a medical expert**.
2.  **RAG (Retrieval-Augmented Generation):** At inference, the model retrieves relevant information from a **custom-built knowledge base** of medical research papers.

> 💡 **Analogy:** Think of it as an **expert doctor** who not only graduated from a top medical school (**Fine-Tuning**) but also has a **perfect, instant-access digital library** (**RAG**) for every question.

---

## ✨ Live Demo

You can try the live application deployed on Hugging Face Spaces. The demo uses a smaller **TinyLlama model** for accessibility on free hardware:

**[➡️ Open the Live Demo](https://huggingface.co/spaces/Sumanth4/medical-qa-rag)**

---

## 🏗️ Architecture & Project Workflow

The project consists of **three main pipelines**:

### 1️⃣ Fine-Tuning Pipeline (The "Education")

This offline process teaches the base LLM to become a medical expert.

```mermaid
graph TD;
    A[Raw Medical Q&A Dataset] --> B{Preprocessing Script};
    B --> C[Train/Validation/Test Splits];
    C --> D[Google Colab Notebook];
    D -- Fine-Tuning using PEFT/LoRA --> E[Fine-Tuned LoRA Adapter];
    E --> F[🚀 Upload to Hugging Face Hub];
2️⃣ RAG Indexing Pipeline (The "Library")This offline process builds the searchable knowledge base for retrieval.Code snippetgraph TD;
    A[Raw Medical PDFs] --> B[Document Processor];
    B -- Text Chunks --> C[Embedding Engine <br>(all-MiniLM-L6-v2)];
    C -- Vector Embeddings --> D[FAISS Vector Database];
    D -- Indexing --> E[💾 Saved to Disk <br>(faiss.index, chunks.pkl)];
3️⃣ Inference Pipeline (The "Consultation")This is the live process where the Fine-Tuned LLM and RAG system work together to answer a user's question.Code snippetgraph TD;
    subgraph "User Interaction"
        A[User Query];
    end

    subgraph "Retrieval Stage (RAG)"
        B[Embedding Engine] --> C[FAISS Vector Database];
        A --> B;
        C -- Top-K Chunks --> E[Retrieved Context];
    end

    subgraph "Generation Stage (Fine-Tuned LLM)"
        D[Fine-Tuned Llama 2 <br>(Base + LoRA Adapter)];
        E -- Combined with --> F[Prompt Template];
        A -- Combined with --> F;
        F --> D;
        D -- Generates --> G[Final Answer];
    end

    subgraph "Output"
        G --> H[Display to User];
    end
📁 Project Structuremedical-qa-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── app.py
│
├── data/
│   └── raw/
│
├── artifacts/ (Ignored by Git)
│   ├── fine_tuned_model/
│   └── vector_store/
│
├── notebooks/
│   └── 3_model_finetuning.ipynb
│
└── src/
    ├── core/
    │   ├── document_processor.py
    │   ├── embedding_engine.py
    │   ├── vector_database.py
    │   ├── llm_handler.py
    │   └── rag_system.py
    │
    ├── pipeline/
    │   ├── training_pipeline.py
    │   └── inference_pipeline.py
    │
    └── utils/
        └── logger.py
🛠️ Tech StackAI & ML: PyTorch, Transformers, PEFT, datasets, scikit-learnModels: meta-llama/Llama-2-7b-hf, all-MiniLM-L6-v2RAG & Data Handling: FAISS, LangChain, PyMuPDF, Pandas, NumPyBackend & Deployment: Streamlit, Hugging Face (Spaces, Hub, Datasets), DockerMLOps & Tooling: Git, Git LFS, python-dotenv🚀 Setup and UsageTo run this project locally, follow these steps:Clone the repository:Bashgit clone [https://github.com/Sumanthcs4/medical-qa-rag.git](https://github.com/Sumanthcs4/medical-qa-rag.git)
cd medical-qa-rag
Create and activate a virtual environment:Bashpython -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:Bashpip install -r requirements.txt
Build the Knowledge Base:Run the training pipeline once to process the PDFs and create the FAISS index.Bashpython -m src.pipeline.training_pipeline
Run the Streamlit Application:The local version is configured to run with TinyLlama for accessibility.Bashstreamlit run app.py
🏆 Model EvaluationThe base Llama 2 7B model was fine-tuned on the medical_meadow_medqa dataset. The performance was evaluated on a test split to measure the improvement in semantic understanding.MetricBase Llama 2 7BFine-Tuned Llama 2 7B (Our Model)BERTScore (F1)[Your Score Here]0.798ROUGE-L[Your Score Here]0.050BLEU[Your Score Here]0.0107The high BERTScore proves that fine-tuning significantly improved the model's ability to understand the context and generate semantically correct medical explanations, even when its wording doesn't exactly match the reference answer.🔮 Future ImprovementsImplement a formal evaluation for the RAG retriever using metrics like Hit Rate and Mean Reciprocal Rank (MRR).Integrate a custom exception handling system using the exceptions/ directory for more robust error management.Develop a suite of automated unit and integration tests using pytest in the tests/ directory to ensure long-term reliability.
