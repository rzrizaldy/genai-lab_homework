import numpy as np
import pandas as pd
from transformers import pipeline
import huggingface_hub, datasets, transformers, os, sys, json, random, re, nltk, time, requests, faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


class LLM_requesters:

    def __init__(self):
        pass 

    def query_llm_model(self, payload, HF_INFERENCE_TOKEN, model_endpoint):
        
    	headers = {
    		"Accept" : "application/json",
    		"Authorization": f"Bearer {HF_INFERENCE_TOKEN}",    # insert your HF inference endpoint token here. Email Sara Kingsley with any questions you have. 
    		"Content-Type": "application/json"
    	}
    	response = requests.post(
            f"{model_endpoint}",
    		headers=headers,
    		json=payload
    	)
    	return response.json()
    
    def get_llm_response(self, model, user_prompt, system_message, task):
        
        huggingface_token= os.getenv("HF_TOKEN")
        
        if not huggingface_token:
            raise RuntimeError("No Hugging Face API token found in the environment variables. Please set 'HUGGINGFACE_TOKEN' in the .env file.")
        
        # Use the pipeline function imported in CELL INDEX: 28
        pipe = pipeline("text-generation", model=model,  token=huggingface_token)  
    
    
        if isinstance(user_prompt, str):
            prompts = [user_prompt]  # Convert the single prompt to a list
    
        responses = []
        
        system_prompt = f"Instruction: perform this {task}."
    
        for prompt in prompts:
            try:
                outputs = pipe(f"{system_prompt} {user_prompt} {system_message}", max_length=500, num_return_sequences=1, truncation=True)
                response = outputs[0]["generated_text"][-1] 
                responses.append(response)
            except Exception as e:
                responses.append(f"Error generating response: {e}")
    
        return responses if len(responses) > 1 else responses[0] 

    

