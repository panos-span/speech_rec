#!/usr/bin/env python3
"""
Data preparation script for USC-TIMIT dataset with Kaldi.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def read_filesets(filesets_dir: Path, dataset_name: str) -> List[str]:
    """
    Read the fileset file containing utterance IDs
    """
    fileset_path = os.path.join(filesets_dir, f"{dataset_name}.txt")
    with open(fileset_path, "r") as f:
        uutids = [line.strip() for line in f if line.strip()]
    return uutids


def read_transcriptions(trans_file: Path) -> List[str]:
    """
    Read the transcription file
    """
    with open(trans_file, "r") as f:
        transcriptions = [line.strip() for line in f if line.strip()]
    return transcriptions


def read_lexicon(lexicon_file: Path) -> Dict[str, str]:
    """
    Read the pronunciation lexicon
    """
    lexicon = {}
    with open(lexicon_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                phones = " ".join(parts[1:])
                lexicon[word] = phones
    return lexicon


def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase and removing special
    characters except apostrophes
    """
    cleaned = "".join(
        c if c.isalnum() or c.isspace() or c == "'" else " "
        for c in text.lower()
    )
    # Replace multiple spaces with a single space
    cleaned = " ".join(cleaned.split())
    return cleaned


def word_to_phones(word: str, lexicon: Dict[str, str]) -> str:
    """
    Convert text to phone sequence using the pronunciation lexicon
    """
    word = word.lower()
    if word in lexicon:
        return lexicon[word]
    else:
        print(f"Word '{word}' not found in lexicon", file=sys.stderr)
        return "spn"


def text_to_phones(text: str, lexicon: Dict[str, str]) -> str:
    """
    Convert text to phone sequence
    """
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    phone_sequence = []

    # Add silence at the beginning
    phone_sequence.append("sil")

    # Convert each word to phones
    for word in words:
        phones = word_to_phones(word, lexicon)
        phone_sequence.append(phones)

    # Add silence at the end
    phone_sequence.append("sil")

    return " ".join(phone_sequence)


def parse_uttid(uutid: str) -> Tuple[str, int]:
    """
    Parse the utterance ID to get speaker ID and sentence ID
    """
    # Format: {speaker}_{sentence_id} (e.g. f1_001)
    parts = uutid.split("_")
    speaker = parts[0]
    sentence_id = int(parts[1])
    return speaker, sentence_id


def generate_kaldi_files(
    uttids: Tuple[str, int],
    wav_dir: Path,
    transcriptions: List[str],
    lexicon: Dict[str, str],
    output_dir: Path,
) -> None:
    """
    Generate Kaldi data files: uttids, utt2spk, wav.scp, text
    """
    # Write uttids
    with open(os.path.join(output_dir, "uttids"), "w") as f:
        for uutid in uttids:
            f.write(f"{uutid}\n")

    # Write utt2spk
    with open(os.path.join(output_dir, "utt2spk"), "w") as f:
        for uutid in uttids:
            speaker, _ = parse_uttid(uutid)
            f.write(f"{uutid} {speaker}\n")

    # Write wav.scp
    with open(os.path.join(output_dir, "wav.scp"), "w") as f:
        for uutid in uttids:
            # Define the path to the WAV file
            wav_path = os.path.join(wav_dir, f"{uutid}.wav")
            wav_path = os.path.abspath(wav_path)

            f.write(f"{uutid} {wav_path}\n")

    # Write text (phoneme transcriptions)
    with open(os.path.join(output_dir, "text"), "w") as f:
        for uutid in uttids:
            _, sentence_id = parse_uttid(uutid)

            # Get the corresponding transcription (0-based indexing)
            if 0 < sentence_id <= len(transcriptions):
                trans = transcriptions[sentence_id - 1]

                # Convert text to phone sequence
                phone_sequence = text_to_phones(trans, lexicon)

                f.write(f"{uutid} {phone_sequence}\n")
            else:
                print(
                    f"Warning: Sentence ID {sentence_id} out of range",
                    file=sys.stderr,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Data preparation for USC-TIMIT dataset with Kaldi"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Base directory containing USC data"
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "dev", "test"],
        required=True,
        help="Dataset to prepare",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Define paths based on the actual file structure
    wav_dir = Path(args.data_dir) / "wav"
    filesets_dir = Path(args.data_dir) / "filesets"
    trans_file = Path(args.data_dir) / "transcriptions.txt"
    lexicon_file = Path(args.data_dir) / "lexicon.txt"

    # Map dataset names to fileset names
    fileset_map = {"train": "training", "dev": "validation", "test": "testing"}

    # Read the fileset (utterance IDs)
    uttids = read_filesets(filesets_dir, fileset_map[args.dataset])

    # Read transcriptions
    transcriptions = read_transcriptions(trans_file)

    # Read lexicon
    lexicon = read_lexicon(lexicon_file)

    # Generate kaldi files
    generate_kaldi_files(
        uttids, wav_dir, transcriptions, lexicon, Path(args.output_dir)
    )

    # Create spk2utt file using Kaldi utility
    os.system(
        f"utils/utt2spk_to_spk2utt.pl {args.output_dir}/utt2spk > {args.output_dir}/spk2utt"
    )

    print(
        f"Data preparation for {args.dataset} dataset completed successfully"
    )


if __name__ == "__main__":
    main()
