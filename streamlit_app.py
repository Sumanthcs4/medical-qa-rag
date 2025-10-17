# streamlit_app.py

import sys, os
# 👇 Add this to make sure Python can find the 'src' folder
sys.path.append(os.path.abspath("src"))

import streamlit as st
from pipeline.inference_pipeline import InferencePipeline

# --- 1. The Caching Function (Crucial for Performance) ---
@st.cache_resource
def load_pipeline():
    """Loads and returns the RAG inference pipeline."""
    try:
        pipeline = InferencePipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize the RAG pipeline: {e}")
        return None

# --- 2. The Main Application UI ---
st.set_page_config(page_title="Medical QA", page_icon="🩺")

st.title("🩺 Medical Question Answering System")
st.markdown(
    "This application uses a fine-tuned Llama 2 model with a RAG pipeline "
    "to answer questions based on a specialized medical knowledge base."
)

with st.spinner(
    "Initializing the AI engine... This may take several minutes on the first run as models are downloaded."
):
    pipeline = load_pipeline()

if pipeline:
    st.success("AI Engine is ready. You may now ask your question.")
    user_query = st.text_input("Enter your medical question here:")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Searching the knowledge base and generating an answer..."):
                try:
                    answer = pipeline.ask_question(user_query)
                    st.write("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
        else:
            st.warning("Please enter a question first.")
else:
    st.error(
        "The AI engine could not be loaded. The application cannot proceed. "
        "Please check the logs on the server for more details."
    )
