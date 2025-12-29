#!/usr/bin/env python3
"""
Split training data into train (90%) and validation (10%) sets
Maintains balanced positive/negative sample ratio
Saves split data as pickle files for reuse in other scripts
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'


def load_training_data():
    """Load all training data (positive and negative tweets)"""
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print("Loading training data...")
    
    X = []
    y = []
    
    # Load positive tweets
    pos_count = 0
    pos_file = os.path.join(project_root, "public_kaggle_files/twitter-datasets/train_pos.txt")
    with open(pos_file, encoding='utf-8') as f:
        for line in f:
            X.append(line.strip())
            y.append(1)
            pos_count += 1
    
    print(f"Loaded {pos_count} positive tweets")
    
    # Load negative tweets
    neg_count = 0
    neg_file = os.path.join(project_root, "public_kaggle_files/twitter-datasets/train_neg.txt")
    with open(neg_file, encoding='utf-8') as f:
        for line in f:
            X.append(line.strip())
            y.append(-1)
            neg_count += 1
    
    print(f"Loaded {neg_count} negative tweets")
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def split_data(X, y, train_size=0.9, random_state=42):
    """
    Split data into training and validation sets
    Maintains balanced positive/negative sample ratio
    """
    print("\nSplitting data into train (90%) and validation (10%)...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=y  # Ensure balanced split
    )
    
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == -1)
    val_pos = np.sum(y_val == 1)
    val_neg = np.sum(y_val == -1)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"  Positive: {train_pos}, Negative: {train_neg}")
    print(f"  Ratio: {train_pos / (train_pos + train_neg):.4f} positive")
    
    print(f"Validation set: {len(X_val)} samples")
    print(f"  Positive: {val_pos}, Negative: {val_neg}")
    print(f"  Ratio: {val_pos / (val_pos + val_neg):.4f} positive")
    
    return X_train, X_val, y_train, y_val


def save_split_data(X_train, X_val, y_train, y_val, output_dir):
    """Save split data as pickle files"""
    print("\nSaving split data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train_texts.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    print("Saved: train_texts.pkl")
    
    with open(os.path.join(output_dir, "train_labels.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    print("Saved: train_labels.pkl")
    
    with open(os.path.join(output_dir, "val_texts.pkl"), "wb") as f:
        pickle.dump(X_val, f)
    print("Saved: val_texts.pkl")
    
    with open(os.path.join(output_dir, "val_labels.pkl"), "wb") as f:
        pickle.dump(y_val, f)
    print("Saved: val_labels.pkl")
    
    print("\nData split completed!")
    print("Files ready for use in other classification scripts")


def main():
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'output')
    
    print("=" * 60)
    print("Data Split: 90% Training + 10% Validation")
    print("=" * 60)
    
    # Load data
    X, y = load_training_data()
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y, train_size=0.9)
    
    # Save split data
    save_split_data(X_train, X_val, y_train, y_val, output_dir)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
