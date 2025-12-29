#!/usr/bin/env python3
"""
Filter vocabulary list to keep only words with frequency >= 5.
This script removes low-frequency words that are likely to be noise or typos.
Input: vocab_full.txt (from build_vocab.py)
Output: vocab_cut.txt - contains only frequent words (one word per line)
"""

import os


def main():
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'output')
    
    print("Filtering vocabulary...")
    
    input_file = os.path.join(output_dir, "vocab_full.txt")
    output_file = os.path.join(output_dir, "vocab_cut.txt")
    min_frequency = 5
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run build_vocab.py first.")
        return
    
    # Read and filter vocabulary
    filtered_words = []
    total_words = 0
    filtered_count = 0
    
    print(f"Reading vocabulary from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_words += 1
            
            # Parse "count word" format
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            try:
                count = int(parts[0])
                word = parts[1]
                
                # Keep only words with frequency >= min_frequency
                if count >= min_frequency:
                    filtered_words.append(word)
                    filtered_count += 1
            except ValueError:
                continue
    
    print(f"Total unique words in input: {total_words}")
    print(f"Words with frequency >= {min_frequency}: {filtered_count}")
    print(f"Words removed (frequency < {min_frequency}): {total_words - filtered_count}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write filtered vocabulary
    print(f"Writing filtered vocabulary to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in filtered_words:
            f.write(word + "\n")
    
    print("Vocabulary filtering completed!")
    print(f"Output saved to: {output_file}")
    print(f"Final vocabulary size: {len(filtered_words)}")


if __name__ == "__main__":
    main()
