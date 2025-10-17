# app.py

import streamlit as st
from src.pipeline.inference_pipeline import InferencePipeline

# --- 1. The Caching Function (Crucial for Performance) ---
# This decorator tells Streamlit to run this function ONLY ONCE.
# It will load your models and keep them in memory, which is vital
# because this process is very slow and resource-intensive.
@st.cache_resource
def load_pipeline():
    """
    Loads and returns the RAG inference pipeline.
    This function is cached to prevent reloading the models on every interaction.
    """
    try:
        pipeline = InferencePipeline()
        return pipeline
    except Exception as e:
        # Display an error message in the app if the pipeline fails to load
        st.error(f"Failed to initialize the RAG pipeline: {e}")
        return None

# --- 2. The Main Application UI ---

# Set a title and an icon for the browser tab
st.set_page_config(page_title="Medical QA", page_icon="🩺")

# Display the main title and a brief description
st.title("🩺 Medical Question Answering System")
st.markdown(
    "This application uses a fine-tuned Llama 2 model with a RAG pipeline "
    "to answer questions based on a specialized medical knowledge base."
)

# Display a status message while the pipeline is loading
with st.spinner("Initializing the AI engine... This may take several minutes on the first run as models are downloaded."):
    pipeline = load_pipeline()

# Only show the main interface if the pipeline has been loaded successfully
if pipeline:
    st.success("AI Engine is ready. You may now ask your question.")
    
    # Create a text input box for the user's question
    user_query = st.text_input("Enter your medical question here:")

    # Create a button that the user will click to get an answer
    if st.button("Get Answer"):
        if user_query:
            # If the user has entered a question, run the pipeline
            with st.spinner("Searching the knowledge base and generating an answer..."):
                try:
                    # This is where you call your powerful backend!
                    answer = pipeline.ask_question(user_query)
                    
                    # Display the final answer
                    st.write("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
        else:
            # If the button is clicked with no text, show a warning
            st.warning("Please enter a question first.")
else:
    st.error(
        "The AI engine could not be loaded. The application cannot proceed. "
        "Please check the logs on the server for more details."
    )