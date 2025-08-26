#!/usr/bin/env python3
"""
build_faiss_index.py

Precomputes embeddings and a FAISS index for a movie dataset.
This script is designed to be run once before deploying the search application.

It performs the following steps:
1. Loads a movie dataset from a CSV file.
2. Cleans and preprocesses the data to create a combined text field for embedding.
3. Uses a SentenceTransformer model to compute embeddings for the text field.
4. Builds a FAISS index for efficient similarity search.
5. Saves the metadata, embeddings, and FAISS index to disk.

Example usage:
    python build_faiss_index.py --input_csv ./movies.csv
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths ---
DEFAULT_CSV_PATH = "./movies.csv"
EMBEDDINGS_PATH = "movie_embeddings.npy"
META_PATH = "movie_meta.parquet"
FAISS_INDEX_PATH = "faiss.index"

# --- Model Configuration ---
# Reduce batch size to avoid CPU OOM when running large models on long texts.
BATCH_SIZE = 8
# Cap the tokenized sequence length to keep memory usage predictable.
MAX_SEQ_LENGTH = 256
# Using the specified open-source embedding model from Hugging Face.
# Ensure this matches the model used in the search app.
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_and_preprocess_movies(csv_path: Path) -> pd.DataFrame:
    """
    Loads movie data from a CSV, preprocesses it, and creates a 'meta_text' field for embedding.

    Args:
        csv_path: Path to the input movies.csv file.

    Returns:
        A pandas DataFrame with preprocessed data and a '__pos' column for indexing.
    """
    if not csv_path.exists():
        logging.error(f"CSV file not found at: {csv_path}")
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    logging.info(f"Loading data from {csv_path}...")
    # Read all columns as strings to prevent dtype issues and fill NaNs.
    df = pd.read_csv(csv_path, low_memory=False, dtype=str).fillna("")
    df = df.reset_index(drop=True)
    # Create a stable position identifier for mapping FAISS results back to metadata.
    df["__pos"] = np.arange(len(df))

    # --- Feature Engineering ---
    # Ensure required columns exist, creating them if they don't.
    required_cols = ["title", "overview", "genres", "cast", "crew", "keywords", "release_date"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
            logging.warning(f"Column '{col}' not found. Added as an empty column.")

    # Safely extract the year from the release_date.
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    # Combine relevant text fields into a single 'meta_text' for a rich embedding.
    # The separator helps the model distinguish between different fields.
    text_fields = ["title", "overview", "genres", "cast", "crew", "keywords"]
    df["meta_text"] = df[text_fields].apply(lambda r: " | ".join(r.astype(str)), axis=1)

    logging.info(f"Loaded and preprocessed {len(df)} movies.")
    return df


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """
    Computes sentence embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts: A list of strings to be embedded.
        model_name: The name of the SentenceTransformer model to use.
        batch_size: The batch size for encoding.

    Returns:
        A numpy array of embeddings.
    """
    # Use CUDA if available for a significant speed-up.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading SentenceTransformer model: {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device)
    # Enforce a reasonable maximum sequence length to avoid excessive memory usage.
    try:
        original_max_len = getattr(model, "max_seq_length", None)
        model.max_seq_length = MAX_SEQ_LENGTH
        logging.info(f"Set model.max_seq_length from {original_max_len} to {model.max_seq_length}")
    except Exception as e:
        logging.warning(f"Could not set max_seq_length: {e}")

    logging.info(f"Computing embeddings for {len(texts)} texts...")
    # normalize_embeddings=True is crucial for using dot product (IP) similarity.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_and_save_faiss_index(embeddings: np.ndarray, output_path: str):
    """
    Builds a FAISS index from embeddings and saves it to a file.

    Args:
        embeddings: The numpy array of embeddings.
        output_path: The path to save the FAISS index file.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    num_embeddings, dim = embeddings.shape
    logging.info(f"Building FAISS index for {num_embeddings} vectors of dimension {dim}...")

    # IndexFlatIP is ideal for dense vectors with normalized embeddings,
    # as it computes the exact Inner Product (equivalent to cosine similarity here).
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logging.info(f"Saving FAISS index to {output_path}...")
    faiss.write_index(index, output_path)


def main(input_csv: str):
    """
    Main function to orchestrate the data loading, embedding, and indexing pipeline.
    """
    start_time = time.time()
    
    # 1. Load and preprocess data
    df = load_and_preprocess_movies(Path(input_csv))
    
    # 2. Compute embeddings
    embeddings = compute_embeddings(df["meta_text"].tolist(), MODEL_NAME, BATCH_SIZE)
    if embeddings.shape[0] != len(df):
        raise RuntimeError("Mismatch between number of embeddings and metadata rows.")

    # 3. Save artifacts
    logging.info(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)

    build_and_save_faiss_index(embeddings, FAISS_INDEX_PATH)

    logging.info(f"Saving metadata to {META_PATH}...")
    # Select columns to save to keep the metadata file lean.
    columns_to_save = ["__pos", "title", "overview", "genres", "year"]
    df[columns_to_save].to_parquet(META_PATH, index=False)

    total_time = time.time() - start_time
    logging.info("--- Pipeline Complete ---")
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    logging.info("Produced files:")
    logging.info(f" - Metadata: {META_PATH}")
    logging.info(f" - Embeddings: {EMBEDDINGS_PATH}")
    logging.info(f" - FAISS Index: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index for movie search.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the input movies CSV file (default: {DEFAULT_CSV_PATH})."
    )
    args = parser.parse_args()
    main(args.input_csv)