import os
from qianfan import Qianfan
from dotenv import load_dotenv
import requests
from Function.sptt import socketio
import json
import time
import random
from Model_Class import DistinguishModel
import torch
import faiss
import numpy as np
from transformers import RobertaTokenizer

load_dotenv()  

access_key = os.getenv("QIANFAN_ACCESS_KEY")
secret_key = os.getenv("QIANFAN_SECRET_KEY")
api_key = os.getenv("QIANFAN_API_KEY")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistinguishModel()

# Load pre-trained model from local file
model.load_state_dict(torch.load('Final_distinguish_model.pt', map_location=device))

model.to(device)
model.eval()  # Set model to evaluation mode




def paraphrase(text, token_count):
    # Check input parameters
    print(token_count)
    if not text or not text.strip():
        print("Warning: Empty text provided to paraphrase function")
        return ""
    
    if not token_count or token_count <= 0:
        print("Warning: Invalid token count")
        token_count = 100  # Set default value
    
    # Modify prompt, explicitly specify output format, avoid returning formatted text
    prompt = f"Please rewrite the following text to be more concise while preserving all key information. All original meanings are retained. Keep sentences coherent. Correct wrong or missing words.Return ONLY the rewritten text without any introduction, prefix or formatting:\n\n{text}"
    
    try:
        # Ollama local API call
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3",  # Replace with your installed model name
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": int(0.9*token_count) + 30
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            if not result:
                print("Warning: Model returned empty response")
            
            # Clean results, remove common prefix text
            cleaned_result = result
            prefixes_to_remove = [
                "Here's a rewritten version of your text:",
                "Here's a rewritten version of the content:",
                "Rewritten text:",
                "Rewrite result:"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_result.startswith(prefix):
                    cleaned_result = cleaned_result[len(prefix):].strip()
            
            return cleaned_result
        else:
            print(f"Error: API returned status code {response.status_code}")
            return f"Error occurred: API status code {response.status_code}"
            
    except Exception as e:
        print(f"Error in paraphrase function: {e}")
        return f"Error occurred: {str(e)}"


# Load tokenizer that matches the model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Text encoding processing
def encode(text):
    # Ensure it's a single string
    if isinstance(text, list):
        text = text[0]
    
    # Tokenize and encode the text
    encoded_input = tokenizer(
        text, 
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    
    # Move tensors to the appropriate device
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Use the model for inference
    with torch.no_grad():
        model.eval()
        _, _, sentence_embedding = model(input_ids, attention_mask)
        
    # Move sentence vector from GPU to CPU and convert to numpy array
    sentence_embedding = sentence_embedding.cpu().numpy()
    
    return sentence_embedding

# Configure parameters
API_KEY = "sk-mdivapssxsagmwawlccmouogbmlpvnelrmiwsgarrryoncfv"  # Please verify key validity


def gen_answer(prompt):
    global model
    
    try:
        # Encode the question as a vector (and normalize)
        embeddings = encode(prompt)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        # Check if index file exists
        if not os.path.exists("my_faiss.index"):
            print("Warning: FAISS index file does not exist, cannot perform similar text retrieval")
            # Use original prompt directly, don't add context
            context_text = ""
        else:
            try:
                index = faiss.read_index("my_faiss.index")
                
                # Try to load text fragments
                try:
                    with open("faiss_chunks.json", "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    print(f"Successfully loaded {len(chunks)} text fragments")
                
                    # Retrieve the 5 most relevant text fragments to the question in FAISS
                    k = min(5, index.ntotal)
                    D, I = index.search(embeddings, k)  # D is similarity score, I is index
                    top_idx = I[0]    # List of indices of most relevant text fragments
                    
                    # Ensure indices don't go out of bounds
                    valid_idx = [i for i in top_idx if i < len(chunks)]
                    top_chunks = [chunks[i] for i in valid_idx]
                    context_text = "\n".join(top_chunks)  # Connect multiple fragments with newlines
                    
                except Exception as e:
                    print(f"Error loading text fragments: {e}")
                    context_text = ""
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                context_text = ""
        
        # Build prompt based on whether context exists
        if context_text:
            prompt_text = f"Answer the user's questions based on the following user conversation history. \ndocument content:\n{context_text}\n\nquestion: {prompt}\n"
        else:
            prompt_text = prompt
            
        # API call section
        url = "https://api.siliconflow.cn/v1/chat/completions"
        print("Prompt sent to API:" + prompt_text)
        
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{
                "role": "user",
                "content": prompt_text
            }],
            "stream": False,
            "temperature": 0.7,
            "frequency_penalty": 0.2,
            "max_tokens": 80
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Send request
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        # Process response
        if response.status_code == 200:
            response_data = response.json()
            
            print("API response successfully received")
            
            # Extract content
            if response_data.get('choices'):
                return response_data['choices'][0]['message']['content']
            else:
                return "API response format abnormal, answer content not found"
        else:
            return f"Request failed, status code: {response.status_code}, error message: {response.text}"
    
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        return f"Sorry, an error occurred while processing your question: {str(e)}"