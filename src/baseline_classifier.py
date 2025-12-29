#!/usr/bin/env python3

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'


EMBEDDING_DIM = 50
RANDOM_SEED = 42


def load_vocab_only():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'
    )
    with open(os.path.join(output_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    return vocab


def init_random_embeddings(vocab, dim=EMBEDDING_DIM):
    np.random.seed(RANDOM_SEED)
    vocab_size = len(vocab)
    embeddings = np.random.normal(scale=0.1, size=(vocab_size, dim))
    return embeddings


def get_tweet_vector(tweet, vocab, embeddings):
    tokens = tweet.lower().split()
    vectors = []

    for t in tokens:
        if t in vocab:
            idx = vocab[t]
            vectors.append(embeddings[idx])

    if not vectors:
        return np.zeros(embeddings.shape[1])

    return np.mean(vectors, axis=0)


def load_split_texts():
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'
    )

    with open(os.path.join(base, "train_texts.pkl"), "rb") as f:
        train_texts = pickle.load(f)
    with open(os.path.join(base, "train_labels.pkl"), "rb") as f:
        y_train = pickle.load(f)

    return train_texts, y_train


def load_validation_texts():
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'
    )

    with open(os.path.join(base, "val_texts.pkl"), "rb") as f:
        val_texts = pickle.load(f)
    with open(os.path.join(base, "val_labels.pkl"), "rb") as f:
        y_val = pickle.load(f)

    return val_texts, y_val


def vectorize_texts(texts, vocab, embeddings):
    print(f"Vectorizing {len(texts)} texts (weak baseline)...")
    return np.array([get_tweet_vector(t, vocab, embeddings) for t in texts])


def train_classifier(X, y):
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_SEED
    )
    clf.fit(X, y)
    return clf


def main():
    print("Loading vocab only (no pretrained embeddings)...")
    vocab = load_vocab_only()
    embeddings = init_random_embeddings(vocab)

    train_texts, y_train = load_split_texts()
    X_train = vectorize_texts(train_texts, vocab, embeddings)

    val_texts, y_val = load_validation_texts()
    X_val = vectorize_texts(val_texts, vocab, embeddings)

    clf = train_classifier(X_train, y_train)

    print("\nEvaluation:")
    print(f"Train accuracy: {accuracy_score(y_train, clf.predict(X_train)):.4f}")
    print(f"Val accuracy:   {accuracy_score(y_val, clf.predict(X_val)):.4f}")


if __name__ == "__main__":
    main()