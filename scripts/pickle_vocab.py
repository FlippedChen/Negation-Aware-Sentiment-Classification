#!/usr/bin/env python3
import pickle
import os


def main():
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'output')
    
    vocab = dict()
    vocab_cut_path = os.path.join(output_dir, "vocab_cut.txt")
    with open(vocab_cut_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    vocab_pkl_path = os.path.join(output_dir, "vocab.pkl")
    with open(vocab_pkl_path, "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
