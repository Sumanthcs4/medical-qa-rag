from transformers import AutoModelCausalLM, AutoTokenizer
from config import LLM_MODEL_PATH
class LlmHandler:
    def __init__(self):
        # Load the tokenizer simply
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        
        # Load the massive model with memory-saving tricks
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            load_in_4bit=True,
            device_map="auto"
        )
    def generate_response(self,query,retrieved_chunks):
        context = "\n---\n".join(retrieved_chunks)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_response = response.replace(prompt, "").strip()
        return cleaned_response