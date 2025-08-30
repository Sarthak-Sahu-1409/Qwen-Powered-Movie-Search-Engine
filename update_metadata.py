# update_metadata.py
import pandas as pd
import pickle
import ast

print("Starting metadata update with director information...")

# --- Configuration ---
MOVIES_CSV_PATH = 'movies.csv'
DATA_SAVE_PATH = 'movies_data.pkl'

# --- Load and Process Data ---
df = pd.read_csv(MOVIES_CSV_PATH, low_memory=False)

# --- MODIFIED: Added 'director' to the required columns ---
required_columns = ['movie_id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'runtime', 'tagline', 'cast', 'director']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CRITICAL ERROR: Your movies.csv file is missing the required column '{col}'.")

df = df[required_columns].copy()
df.rename(columns={'movie_id': 'id'}, inplace=True)

# Use the original, less strict cleaning logic to match your FAISS index
df.dropna(subset=['id', 'title', 'overview'], inplace=True)
df['id'] = pd.to_numeric(df['id'], errors='coerce')
df.dropna(subset=['id'], inplace=True)

# Simplified genre handling
df['genres'] = df['genres'].fillna('N/A').astype(str)

# Robust cast parser
def parse_cast(cast_str, limit=3):
    if pd.isna(cast_str) or not isinstance(cast_str, str):
        return 'N/A'
    try:
        cast_list = ast.literal_eval(cast_str)
        if isinstance(cast_list, list) and cast_list:
            top_actors = [actor.get('name') for actor in cast_list if isinstance(actor, dict) and actor.get('name')]
            return ', '.join(top_actors[:limit]) if top_actors else 'N/A'
    except (ValueError, SyntaxError):
        return 'N/A'
    return 'N/A'
df['cast'] = df['cast'].apply(parse_cast)

# Parse other columns
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_year'] = df['release_year'].fillna(0).astype(int)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(0).astype(int)
df['tagline'] = df['tagline'].fillna('').astype(str)
# --- NEW: Process director column ---
df['director'] = df['director'].fillna('N/A').astype(str)


# Create the text used for reranking
df['search_text'] = "Title: " + df['title'].astype(str) + ". Overview: " + df['overview'].astype(str)

# Reset index to ensure the final order is correct
df.reset_index(drop=True, inplace=True)

# --- Save the Updated and Synced DataFrame ---
with open(DATA_SAVE_PATH, 'wb') as f:
    pickle.dump(df, f)

print(f"✅ Successfully created a synced '{DATA_SAVE_PATH}' with director information.")