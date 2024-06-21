import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

model_name = None

def set_model_name(embed_model):
    global model_name
    
    model_name = embed_model

def create_vector_db():
    global model_name

    df = pd.read_csv("./dataset/vector-database/toxic_chat_unifying_category.csv")

    model = SentenceTransformer(model_name, trust_remote_code=True)

    if model_name is None:
        raise ValueError("Model name has not been set. Please call set_model_name() first.")

    # Create embeddings for the abnormal prompts
    df["embedding"] = df["user_input"].apply(lambda x: model.encode(x))

    # Convert embeddings to a list for later use
    embeddings = np.vstack(df["embedding"].values).astype('float32')

    # Convert list of embeddings to numpy array
    faiss.normalize_L2(embeddings)

    # Initialize the Faiss index
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    # Save metadata for reference
    metadata = df[["category", "user_input", "model_output"]].to_dict(orient="records")

    return index, model, metadata

def find_most_similar(input_sentence, model, index, metadata, top_k=1):
    input_embedding = model.encode(input_sentence).astype('float32').reshape(1, -1)

    faiss.normalize_L2(input_embedding)

    distances, indices = index.search(input_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        result = {
            f"top_{i+1}_category": metadata[idx]["category"],
            f"top_{i+1}_prompt": metadata[idx]["user_input"],
            f"top_{i+1}_output": metadata[idx]["model_output"],
            f"top_{i+1}_similarity_score": distances[0][i]
        }
        
        results.append(result)

    return results