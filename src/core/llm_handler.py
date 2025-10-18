# In src/core/llm_handler.py
import sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config import LLM_MODEL_NAME
from src.utils.logger import setup_logger

logger = setup_logger()

class LlmHandler:
    def __init__(self):
        logger.info(f"Loading model: {LLM_MODEL_NAME}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        logger.info("LLM Handler initialized successfully.")

    def generate_response(self, query, retrieved_chunks):
        context = "\n---\n".join(retrieved_chunks)
        prompt = f"""<|system|>
        You are a helpful medical assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say you don't know.</s>
        <|user|>
        Context:
        {context}

        Question:
        {query}</s>
        <|assistant|>
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the response to get only the assistant's part
        assistant_response = response.split("<|assistant|>")[-1].strip()
        return assistant_response