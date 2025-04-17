"""
MhRDTech AI - Persian Language Model
Created by Mohammad Javad Malekan
Location: Passau, Bayern, Germany
"""

import os
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import wandb

# Configure logging
logging.basicConfig(
    filename='logs/api_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="MhRDTech AI",
    description="Persian Language Model API by Mohammad Javad Malekan",
    version="1.0.0"
)

# Environment variables
API_KEY = os.getenv("XAI_API_KEY")
MODEL_PATH = "models/parsbert-finetuned"

# Model initialization
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    wandb.init(project="persian-nlp-xai")

@app.post("/generate")
async def generate_text(prompt: str):
    try:
        model, tokenizer = load_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return {"error": str(e)}