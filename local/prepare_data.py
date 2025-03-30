#!/usr/bin/env python3
# prepare_data.py - Prepares USC-TIMIT data for Kaldi speech recognition
# Place this in egs/usc/local/

import os
import re
import sys

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def clean_text(text):
    """Convert text to lowercase and remove special characters except apostrophes"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_lexicon(lexicon_path):
    """Load the phoneme lexicon from file, handling tab-separated entries"""
    lexicon = {}
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Handle tab-separated entries
            if '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0].lower().strip()
                    # Remove variant markers like (1) from words
                    word = re.sub(r'\([0-9]+\)', '', word).strip()
                    phonemes = parts[1].strip().split()
                    lexicon[word] = phonemes
            else:
                # Fallback to space-separated entries
                parts = line.strip().split()
                if len(parts) > 1:
                    word = parts[0].lower()
                    phonemes = parts[1:]
                    lexicon[word] = phonemes
    
    print(f"Loaded {len(lexicon)} words in lexicon")
    return lexicon

def word_to_phonemes(word, lexicon):
    """Convert a word to its phoneme sequence using the lexicon"""
    # Clean the word - remove special characters except apostrophes
    clean_word = re.sub(r'[^\w\']', '', word.lower())
    
    # Try the word as is
    if clean_word in lexicon:
        return lexicon[clean_word]
    
    # Try without apostrophes if present
    if "'" in clean_word:
        clean_word_no_apos = clean_word.replace("'", "")
        if clean_word_no_apos in lexicon:
            return lexicon[clean_word_no_apos]
    
    # If it's a number, try to synthesize phonemes
    if clean_word.isdigit():
        # For now, return a placeholder
        print(f"Warning: Numeric word '{clean_word}' not properly handled")
        return []
    
    # If word not found, report it and return empty list
    print(f"Warning: Word '{clean_word}' not found in lexicon!")
    return []

def load_transcriptions(trans_path):
    """Load transcriptions from file"""
    with open(trans_path, 'r', encoding='utf-8') as f:
        transcriptions = [line.strip() for line in f]
    print(f"Loaded {len(transcriptions)} transcriptions")
    return transcriptions

def process_fileset(fileset_path, output_dir, transcriptions, lexicon, kaldi_root):
    """Process a fileset (train, dev, test) and create required files"""
    # Ensure output directory exists
    create_directory(output_dir)
    
    # Read file list
    with open(fileset_path, 'r') as f:
        utterance_ids = [line.strip() for line in f]
    
    print(f"Processing {len(utterance_ids)} utterances for {os.path.basename(output_dir)}")
    
    # Create uttids file
    with open(os.path.join(output_dir, 'uttids'), 'w') as f:
        for utt_id in utterance_ids:
            f.write(f"{utt_id}\n")
    
    # Create utt2spk file
    with open(os.path.join(output_dir, 'utt2spk'), 'w') as f:
        for utt_id in utterance_ids:
            speaker_id = utt_id.split('_')[0]  # Extract speaker ID (e.g., 'm1' from 'm1_001')
            f.write(f"{utt_id} {speaker_id}\n")
    
    # Create wav.scp file
    with open(os.path.join(output_dir, 'wav.scp'), 'w') as f:
        for utt_id in utterance_ids:
            wav_path = os.path.join(kaldi_root, 'egs', 'usc', 'wav', f"{utt_id}.wav")
            abs_wav_path = os.path.abspath(wav_path)
            f.write(f"{utt_id} {abs_wav_path}\n")
    
    # Create text file with phoneme transcriptions
    with open(os.path.join(output_dir, 'text'), 'w') as f:
        for utt_id in utterance_ids:
            # Extract speaker and utterance number
            parts = utt_id.split('_')
            speaker_id = parts[0]
            utt_num = int(parts[1])
            
            # Get the transcript index (adjusting for missing utterances in m1)
            transcript_idx = utt_num - 1
            if speaker_id == 'm1' and utt_num > 235:
                transcript_idx -= 5  # Adjust for missing utterances 231-235
            
            # Get and clean the transcript
            if transcript_idx < len(transcriptions):
                transcript = transcriptions[transcript_idx]
                cleaned_text = clean_text(transcript)
                words = cleaned_text.split()
                
                # Build phoneme sequence with a single sil at beginning and end
                all_phonemes = []
                for word in words:
                    phonemes = word_to_phonemes(word, lexicon)
                    all_phonemes.extend(phonemes)
                
                # Construct the final phoneme sequence with single sil markers
                phoneme_sequence = "sil " + " ".join(all_phonemes) + " sil"
                
                # Write to file
                f.write(f"{utt_id} {phoneme_sequence}\n")
            else:
                print(f"Warning: No transcript found for {utt_id} (index {transcript_idx})")
    
    print(f"Created Kaldi data files for {os.path.basename(output_dir)}")

def main():
    # Define paths
    kaldi_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    usc_dir = os.path.join(kaldi_root, 'egs', 'usc')
    
    print(f"Working with Kaldi root: {kaldi_root}")
    print(f"USC directory: {usc_dir}")
    
    # Paths to key files
    transcriptions_path = os.path.join(usc_dir, 'transcriptions.txt')
    # Check for alternative file names
    if not os.path.exists(transcriptions_path):
        alt_path = os.path.join(usc_dir, 'transcription.txt')
        if os.path.exists(alt_path):
            print(f"Using alternative transcription file: {alt_path}")
            transcriptions_path = alt_path
    
    lexicon_path = os.path.join(usc_dir, 'lexicon.txt')
    filesets_dir = os.path.join(usc_dir, 'filesets')
    
    # Check if files exist
    for file_path in [transcriptions_path, lexicon_path, filesets_dir]:
        if not os.path.exists(file_path):
            print(f"Error: Required file/directory not found: {file_path}")
            sys.exit(1)
    
    # Load transcriptions and lexicon
    print(f"Loading transcriptions from: {transcriptions_path}")
    transcriptions = load_transcriptions(transcriptions_path)
    
    print(f"Loading lexicon from: {lexicon_path}")
    lexicon = load_lexicon(lexicon_path)
    
    # Process each dataset (train, dev, test)
    print("Processing datasets...")
    datasets = {
        'train': 'training.txt',
        'dev': 'validation.txt',
        'test': 'testing.txt'
    }
    
    for dataset_name, fileset_name in datasets.items():
        fileset_path = os.path.join(filesets_dir, fileset_name)
        if not os.path.exists(fileset_path):
            print(f"Warning: Fileset not found: {fileset_path}")
            continue
        
        output_dir = os.path.join(usc_dir, 'data', dataset_name)
        process_fileset(fileset_path, output_dir, transcriptions, lexicon, kaldi_root)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()