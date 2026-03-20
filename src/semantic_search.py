from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.data_loader import load_data

df = load_data("/Users/jaimeet/Documents/Movie Recommender/data/netflix_cleaned.csv")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded successfully")

def create_embeddings(df):
    content_embedding = embedding_model.encode(
        df['overview'].tolist(),
        convert_to_numpy=True
    ).astype('float32')
    return content_embedding

def similarity_search(query,df,content_embedding,top_k=2):
    if isinstance(query,str):
        query_embedding = embedding_model.encode(
            query,
            convert_to_numpy=True
        ).astype('float32')
    else:
        query_embedding = query
    scores = cosine_similarity([query_embedding], content_embedding)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['scores'] = scores[top_indices]
    return results