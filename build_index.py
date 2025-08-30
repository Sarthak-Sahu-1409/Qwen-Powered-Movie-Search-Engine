import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm # MODIFICATION: Import the tqdm library

# --- Configuration ---
MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
MOVIES_CSV_PATH = 'movies.csv'
INDEX_SAVE_PATH = 'movies.index'
DATA_SAVE_PATH = 'movies_data.pkl'
BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load and Preprocess Data (No Changes Here) ---
print("Step 1: Loading and preprocessing movie data...")
# (The data loading and preprocessing code remains exactly the same)
try:
    df = pd.read_csv(MOVIES_CSV_PATH, low_memory=False)
except FileNotFoundError:
    print(f"Error: '{MOVIES_CSV_PATH}' not found. Please download the dataset.")
    exit()
if 'movie_id' not in df.columns:
    print("Error: Column 'movie_id' not found. Please check your CSV.")
    exit()
df = df[['movie_id', 'title', 'overview', 'genres']].copy()
df.dropna(subset=['movie_id', 'title', 'overview'], inplace=True)
def parse_genres(genres_str):
    if not isinstance(genres_str, str) or genres_str.strip() == '': return ''
    try:
        genres_list = json.loads(genres_str.replace("'", '"'))
        return ', '.join([g['name'] for g in genres_list])
    except (json.JSONDecodeError, TypeError): return ''
df['genres'] = df['genres'].apply(parse_genres)
df['search_text'] = df['title'].astype(str) + '. Overview: ' + df['overview'].astype(str) + '. Genres: ' + df['genres'].astype(str)
df.reset_index(drop=True, inplace=True)
print("Data preprocessing complete.")


# --- 2. Generate Embeddings ---
print(f"\nStep 2: Generating embeddings with {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
model.eval()

def mean_pool(last_hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

all_embeddings = []
# MODIFICATION: Wrap the range object with tqdm() to create a progress bar
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Encoding Batches"):
    batch_texts = df['search_text'][i:i+BATCH_SIZE].tolist()
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    batch_embeddings = mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
    all_embeddings.append(batch_embeddings.cpu().numpy())
    # The old print statement is no longer needed

embeddings = np.vstack(all_embeddings)
print("Embeddings generated successfully.")
print(f"Shape of embeddings matrix: {embeddings.shape}")


# --- 3. Build FAISS Index & 4. Save Artifacts (No Changes Here) ---
print("\nStep 3: Building FAISS index...")
# (Rest of the file is unchanged)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings, dtype=np.float32))
print(f"FAISS index built. Total vectors in index: {index.ntotal}")

print("\nStep 4: Saving index and processed data...")
faiss.write_index(index, INDEX_SAVE_PATH)
with open(DATA_SAVE_PATH, 'wb') as f:
    pickle.dump(df, f)
print("Processed data saved.")

print("\n🎉 Indexing complete! You can now run the application.")