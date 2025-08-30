import os
import faiss
import pickle
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import cohere
from dotenv import load_dotenv

class SearchEngine:
    def __init__(self):
        # ... (initialization is unchanged) ...
        print("Initializing Search Engine...")
        load_dotenv()
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key: raise ValueError("COHERE_API_KEY not found")
        self.model_name = 'Qwen/Qwen3-Embedding-0.6B'
        self.index_path = 'movies.index'
        self.data_path = 'movies_data.pkl'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        print(f"Loading movie data from {self.data_path}...")
        with open(self.data_path, 'rb') as f:
            self.df = pickle.load(f)
        print(f"Loading embedding model '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        self.cohere_client = cohere.Client(cohere_api_key)
        print(f"Search Engine Initialized Successfully on device: {self.device}.")

    def _mean_pool(self, last_hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_query(self, query: str):
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._mean_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy().astype(np.float32)

    def search(self, query: str, top_k: int = 3, genre_filter: str = None, runtime_filter: int = None, director_filter: str = None):
        if not query:
            return []

        query_embedding = self._encode_query(query)
        num_candidates = 500
        distances, indices = self.index.search(query_embedding, num_candidates)
        
        candidate_df = self.df.iloc[indices[0]].copy()

        # Filtering logic
        if genre_filter:
            candidate_df = candidate_df[candidate_df['genres'].str.contains(genre_filter, case=False, na=False)]
        if runtime_filter:
            candidate_df = candidate_df[candidate_df['runtime'] <= runtime_filter]
        if director_filter:
            candidate_df = candidate_df[candidate_df['director'].str.contains(director_filter, case=False, na=False)]

        if candidate_df.empty:
            return []

        # Rerank the Filtered Results with Cohere
        docs_for_rerank = candidate_df['search_text'].tolist()
        try:
            reranked_hits = self.cohere_client.rerank(
                model='rerank-english-v3.0',
                query=query,
                documents=docs_for_rerank,
                top_n=top_k
            )
            # Ensure reranked_hits is not None
            if not reranked_hits:
                return []
                
        except cohere.errors.CohereAPIError as e:
            print(f"Cohere API Error: {e}")
            return candidate_df.head(top_k).to_dict(orient='records')
        
        final_results = []
        for hit in reranked_hits.results:
            movie_info = candidate_df.iloc[hit.index].to_dict()
            movie_info['rerank_score'] = hit.relevance_score
            final_results.append(movie_info)

        return final_results