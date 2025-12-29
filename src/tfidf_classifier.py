#!/usr/bin/env python3

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from glove_template import train_glove

os.environ['PYTHONIOENCODING'] = 'utf-8'


def load_embeddings_and_vocab():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    with open(os.path.join(output_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(embeddings_path):
        print("Loading frozen embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        print("Training GloVe (first time only)...")
        embeddings = train_glove()
        print("Embeddings frozen for future use")
    
    return embeddings, vocab


def load_split_texts():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    with open(os.path.join(output_dir, "train_texts.pkl"), "rb") as f:
        train_texts = pickle.load(f)
    with open(os.path.join(output_dir, "train_labels.pkl"), "rb") as f:
        y_train = pickle.load(f)
    return train_texts, y_train


def load_validation_texts():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    with open(os.path.join(output_dir, "val_texts.pkl"), "rb") as f:
        val_texts = pickle.load(f)
    with open(os.path.join(output_dir, "val_labels.pkl"), "rb") as f:
        y_val = pickle.load(f)
    return val_texts, y_val


def compute_tfidf_weights(texts):
    print("Computing TF-IDF weights...")
    tfidf = TfidfVectorizer(max_features=50000, min_df=5)
    tfidf.fit(texts)
    return tfidf


def vectorize_texts_tfidf(texts, vocab, embeddings, tfidf_vectorizer, cache_file=None):
    print(f"Vectorizing {len(texts)} texts with TF-IDF weights...")
    tfidf_matrix = tfidf_vectorizer.transform(texts)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    vocab_indices = []
    valid_features = []
    
    for i, feature in enumerate(feature_names):
        if feature in vocab:
            idx = vocab[feature]
            if idx < len(embeddings):
                vocab_indices.append(idx)
                valid_features.append(i)
    
    vocab_indices = np.array(vocab_indices)
    valid_features = np.array(valid_features)
    
    if len(vocab_indices) == 0:
        X = np.zeros((len(texts), embeddings.shape[1]))
    else:
        embedding_subset = embeddings[vocab_indices]
        tfidf_subset = tfidf_matrix[:, valid_features]
        
        X = tfidf_subset @ embedding_subset
        
        row_sums = np.array(tfidf_subset.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        X = X / row_sums[:, np.newaxis]
    
    return X


def train_classifier(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    return clf


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    
    print("Loading embeddings and vocab...")
    embeddings, vocab = load_embeddings_and_vocab()
    
    print("Loading training data...")
    train_texts, y_train = load_split_texts()
    
    print("Training TF-IDF vectorizer...")
    tfidf = compute_tfidf_weights(train_texts)
    
    X_train = vectorize_texts_tfidf(train_texts, vocab, embeddings, tfidf)
    
    print("Loading validation data...")
    val_texts, y_val = load_validation_texts()
    X_val = vectorize_texts_tfidf(val_texts, vocab, embeddings, tfidf)
    
    print("Training classifier...")
    clf = train_classifier(X_train, y_train)
    
    print("\nEvaluation:")
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    
    return train_acc, val_acc


if __name__ == "__main__":
    train_acc, val_acc = main()
