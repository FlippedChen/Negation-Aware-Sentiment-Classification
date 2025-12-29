#!/usr/bin/env python3
from scipy.sparse import coo_matrix
import numpy as np
import pickle
import os


def main():
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'output')
    
    # Load vocabulary
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    # Load training texts (IMPORTANT: only training set, not validation)
    print("Loading training texts for co-occurrence computation...")
    train_texts_path = os.path.join(output_dir, "train_texts.pkl")
    
    if not os.path.exists(train_texts_path):
        print(f"Error: {train_texts_path} not found!")
        print("Please run split_data.py first to generate training/validation splits.")
        return
    
    with open(train_texts_path, "rb") as f:
        train_texts = pickle.load(f)
    
    print(f"Computing co-occurrence matrix from {len(train_texts)} training texts...")

    data, row, col = [], [], []
    counter = 1
    
    for line in train_texts:
        # Tokenize and map to vocabulary indices
        tokens = [vocab.get(t, -1) for t in line.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        
        # Build co-occurrence pairs
        for t in tokens:
            for t2 in tokens:
                data.append(1)
                row.append(t)
                col.append(t2)

        if counter % 10000 == 0:
            print(f"  Processed {counter}/{len(train_texts)} texts...")
        counter += 1
    
    print("Building sparse co-occurrence matrix...")
    cooc = coo_matrix((data, (row, col)))
    print("Summing duplicates (this can take a while)...")
    cooc.sum_duplicates()
    
    cooc_path = os.path.join(output_dir, "cooc.pkl")
    print(f"Saving co-occurrence matrix to {cooc_path}...")
    with open(cooc_path, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Co-occurrence matrix computed from training data only!")
    print(f"Shape: {cooc.shape}, Non-zero entries: {cooc.nnz}")


if __name__ == "__main__":
    main()
