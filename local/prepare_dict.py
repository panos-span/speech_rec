#!/usr/bin/env python3
# prepare_dict.py - Creates dictionary files for language model
# Place this in egs/usc/local/

import os
import sys

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def extract_phonemes(lexicon_file):
    """Extract all unique phonemes from the lexicon"""
    phonemes = set()
    
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Handle tab-separated entries
            if '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    phones = parts[1].strip().split()
                    phonemes.update(phones)
            else:
                # Fallback to space-separated entries
                parts = line.strip().split()
                if len(parts) > 1:
                    phonemes.update(parts[1:])
    
    # Ensure 'sil' is in the phoneme set
    phonemes.add('sil')
    
    return sorted(list(phonemes))

def create_dict_files(dict_dir, phonemes):
    """Create dictionary files required for language model preparation"""
    
    # Create silence_phones.txt
    with open(os.path.join(dict_dir, 'silence_phones.txt'), 'w') as f:
        f.write("sil\n")
    print("Created silence_phones.txt")
    
    # Create optional_silence.txt
    with open(os.path.join(dict_dir, 'optional_silence.txt'), 'w') as f:
        f.write("sil\n")
    print("Created optional_silence.txt")
    
    # Create nonsilence_phones.txt (ensure 'sil' is excluded)
    with open(os.path.join(dict_dir, 'nonsilence_phones.txt'), 'w') as f:
        for phone in phonemes:
            if phone != 'sil':
                f.write(f"{phone}\n")
    print("Created nonsilence_phones.txt")
    
    # Create lexicon.txt (1-1 mapping for phone recognition)
    with open(os.path.join(dict_dir, 'lexicon.txt'), 'w') as f:
        # Add silence
        f.write("sil sil\n")
        
        # Add all other phonemes
        for phone in phonemes:
            if phone != 'sil':
                f.write(f"{phone} {phone}\n")
    print("Created lexicon.txt")
    
    # Create empty extra_questions.txt
    with open(os.path.join(dict_dir, 'extra_questions.txt'), 'w') as f:
        pass
    print("Created empty extra_questions.txt")

def create_lm_train_text(data_dir, dict_dir):
    """Create lm_train.text files with <s> and </s> markers"""
    
    for subset in ['train', 'dev', 'test']:
        input_file = os.path.join(data_dir, subset, 'text')
        output_file = os.path.join(dict_dir, f'lm_{subset}.text')
        
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
            
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    uttid, text = parts
                    f_out.write(f"{uttid} <s> {text} </s>\n")
        
        print(f"Created lm_{subset}.text with sentence markers")

def main():
    # Define paths
    kaldi_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    usc_dir = os.path.join(kaldi_root, 'egs', 'usc')
    data_dir = os.path.join(usc_dir, 'data')
    dict_dir = os.path.join(data_dir, 'local', 'dict')
    
    # Create directory if it doesn't exist
    create_directory(dict_dir)
    
    # Get lexicon file path
    lexicon_path = os.path.join(usc_dir, 'lexicon.txt')
    
    if not os.path.exists(lexicon_path):
        print(f"Error: Lexicon file not found at {lexicon_path}")
        sys.exit(1)
    
    # Extract phonemes from lexicon
    print(f"Extracting phonemes from lexicon: {lexicon_path}")
    phonemes = extract_phonemes(lexicon_path)
    print(f"Found {len(phonemes)} unique phonemes")
    
    # Create dictionary files
    create_dict_files(dict_dir, phonemes)
    
    # Create lm_train.text files
    create_lm_train_text(data_dir, dict_dir)
    
    print("Dictionary preparation completed successfully.")

if __name__ == "__main__":
    main()