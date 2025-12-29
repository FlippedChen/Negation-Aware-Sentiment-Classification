#!/usr/bin/env python3
"""
Build vocabulary list from all training data.
This script extracts all unique words from the training texts and counts their frequency.
Note: Uses ALL original training data (not split) to build a complete vocabulary.
This is acceptable because vocabulary construction is a preprocessing step.
Output: vocab_full.txt - contains word frequencies in format: "count word"
"""

from collections import Counter
import os


def main():
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'output')
    
    print("Building vocabulary from training data...")
    
    # Initialize counter for word frequencies
    word_counter = Counter()
    
    # Process both positive and negative training files
    training_files = [
        os.path.join(project_root, "public_kaggle_files/twitter-datasets/train_pos.txt"),
        os.path.join(project_root, "public_kaggle_files/twitter-datasets/train_neg.txt")
    ]
    
    total_tweets = 0
    
    for filepath in training_files:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
            
        print(f"Processing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                # Filter out empty tokens
                tokens = [t for t in tokens if t]
                word_counter.update(tokens)
                total_tweets += 1
                
                if total_tweets % 100000 == 0:
                    print(f"  Processed {total_tweets} tweets...")
    
    print(f"Total tweets processed: {total_tweets}")
    print(f"Unique words found: {len(word_counter)}")
    
    # Write vocabulary with frequencies in descending order
    print("Writing vocabulary to vocab_full.txt...")
    vocab_full_path = os.path.join(output_dir, "vocab_full.txt")
    with open(vocab_full_path, 'w', encoding='utf-8') as f:
        for word, count in word_counter.most_common():
            f.write(f"{count} {word}\n")
    
    print("Vocabulary building completed!")
    print(f"Output saved to: {vocab_full_path}")


if __name__ == "__main__":
    main()
