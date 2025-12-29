#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import os


def train_glove(nmax=100, embedding_dim=50, eta=0.05, alpha=0.75, epochs=10):
    # Get paths relative to project root
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, 'output')
    
    print("Loading cooccurrence matrix...")
    cooc_path = os.path.join(output_dir, "cooc.pkl")
    with open(cooc_path, "rb") as f:
        cooc = pickle.load(f)
    print(f"{cooc.nnz} nonzero entries")

    vocab_size = cooc.shape[0]

    print("Initializing embeddings and biases...")
    xs = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    ys = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    bx = np.zeros(vocab_size)
    by = np.zeros(vocab_size)

    rows = cooc.row
    cols = cooc.col
    data = cooc.data

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        order = np.random.permutation(len(data))

        for k in order:
            i = rows[k]
            j = cols[k]
            xij = data[k]

            logX = np.log(xij)
            fx = min(1.0, (xij / nmax) ** alpha)

            xi_old = xs[i].copy()
            yj_old = ys[j].copy()

            diff = np.dot(xi_old, yj_old) + bx[i] + by[j] - logX
            grad = fx * diff

            xs[i] -= eta * grad * yj_old
            ys[j] -= eta * grad * xi_old
            bx[i] -= eta * grad
            by[j] -= eta * grad

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, xs)
    print(f"Embeddings saved to {embeddings_path}")
    return xs

def main():
    train_glove()


if __name__ == "__main__":
    main()
