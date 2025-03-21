# Initialize the tokenizer and model
from transformers import RobertaTokenizer
import torch
import torch.nn.functional as F
import os
import numpy as np
import faiss
from pymongo import MongoClient
import os
from Model_Class import DistinguishModel
from transformers import RobertaTokenizer
import torch
import torch.nn.functional as F
import json



def split_text_into_sentences(text):
    """
    Split text into a list of sentences by period
    
    Parameters:
        text (str): The text string to be split
    
    Returns:
        list: A list of sentences, each element is a sentence
    """
    # Split text by period
    sentences = text.split('.')
    
    # Clean sentences, remove whitespace and filter out empty sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Check if sentence is not empty
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def distinguish_data(text):
    texts = split_text_into_sentences(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = DistinguishModel()
    
    # Load pre-trained model from local file
    model.load_state_dict(torch.load('Final_distinguish_model.pt', map_location=device))
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Store filtered sentence vectors
    filtered_sentence_vectors = []
    # Store filtered sentences
    filtered_sentences = []
    
    # Process each text in the list
    with torch.no_grad():  # No need to track gradients during inference
        for sentence in texts:  # Note that we use texts here, consistent with the function beginning
            # Tokenize the input
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move inputs to the same device as the model
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Forward pass through the model
            _, classification_logits, embedding = model(input_ids, attention_mask)
            
            # Convert logits to binary prediction (0 or 1)
            prediction = (torch.sigmoid(classification_logits) > 0.9).int().item()
            
            # Only store sentence vectors when prediction is 1
            if prediction == 1:
                filtered_sentence_vectors.append(embedding.cpu().numpy()[0])
                filtered_sentences.append(sentence)  # Save filtered sentence text
    
    # Convert filtered sentence vectors to numpy array
    filtered_sentence_vectors = np.array(filtered_sentence_vectors)
    
    # Check if there are new vectors to add
    if len(filtered_sentence_vectors) > 0:
        index_file = "my_faiss.index"
        chunks_file = "faiss_chunks.json"
        
        # Read existing chunks file (if it exists)
        existing_chunks = []
        if os.path.exists(chunks_file):
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    existing_chunks = json.load(f)
                print(f"Loaded existing text chunks file with {len(existing_chunks)} chunks")
            except Exception as e:
                print(f"Error loading text chunks file: {e}")
                print("Creating new text chunks file")
        
        # Check if index file already exists
        if os.path.exists(index_file):
            # Load existing index
            index = faiss.read_index(index_file)
            print(f"Loaded existing index with {index.ntotal} vectors")
            
            # Ensure vector count matches text chunk count
            if index.ntotal != len(existing_chunks):
                print(f"Warning: Vector count ({index.ntotal}) doesn't match text chunk count ({len(existing_chunks)})")
            
            # Normalize new vectors
            faiss.normalize_L2(filtered_sentence_vectors)
            
            # Add new vectors to the index
            index.add(filtered_sentence_vectors)
            print(f"Added {len(filtered_sentence_vectors)} new vectors, now total: {index.ntotal}")
            
            # Add new text chunks to existing list
            existing_chunks.extend(filtered_sentences)
            print(f"Now have {len(existing_chunks)} text chunks")
        else:
            # Create new index
            vec_dim = filtered_sentence_vectors.shape[1]
            index = faiss.IndexFlatIP(vec_dim)
            
            # Normalize vectors
            faiss.normalize_L2(filtered_sentence_vectors)
            
            # Add vectors to index
            index.add(filtered_sentence_vectors)
            print(f"Created new index with {len(filtered_sentence_vectors)} vectors")
            
            # Use new text chunks list
            existing_chunks = filtered_sentences
            print(f"Created new text chunks list with {len(existing_chunks)} chunks")
        
        # Save updated index
        faiss.write_index(index, index_file)
        print(f"Index saved to {index_file}")
        
        # Save updated text chunks
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
        print(f"Text chunks saved to {chunks_file}")
    else:
        print("No new vectors to add")

