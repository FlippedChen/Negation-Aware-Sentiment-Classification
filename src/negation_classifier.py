#!/usr/bin/env python3

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from glove_template import train_glove

os.environ['PYTHONIOENCODING'] = 'utf-8'

NEGATIONS = {"not", "no", "never", "n't", "dont", "don't", "didn't", "can't", "won't"}
NEGATION_WINDOW = 3


def load_embeddings_and_vocab():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'
    )

    with open(os.path.join(output_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
    else:
        embeddings = train_glove()

    return embeddings, vocab


def get_tweet_vector(tweet, vocab, embeddings):
    tokens = tweet.lower().split()

    vectors = []
    negate_count = 0

    for t in tokens:
        if t in NEGATIONS:
            negate_count = NEGATION_WINDOW
            continue

        if t in vocab:
            idx = vocab[t]
            if idx < len(embeddings):
                v = embeddings[idx]

                if negate_count > 0:
                    v = -1.5 * v
                    negate_count -= 1

                vectors.append(v)

    if not vectors:
        dim = embeddings.shape[1]
        return np.zeros(2 * dim)

    vectors = np.stack(vectors)

    mean_pool = vectors.mean(axis=0)
    max_pool = vectors.max(axis=0)

    return np.concatenate([mean_pool, max_pool])


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
    print(f"Vectorizing {len(texts)} texts...")
    return np.array([get_tweet_vector(t, vocab, embeddings) for t in texts])


def train_classifier(X, y):
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X, y)
    return clf


def main():
    embeddings, vocab = load_embeddings_and_vocab()

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