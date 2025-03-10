#!/usr/bin/env python3
"""
Script to analyze and visualize the phoneme recognition results from Kaldi.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def parse_per_from_logs(logs_file: Path) -> Optional[float]:
    """
    Parse phone error rate (PER) from the logs
    """
    with open(logs_file, "r") as f:
        content = f.read()

    # Regu;ar expression to find PER (shown as WER in Kaldi output)
    per = re.search(r"%WER\s+(\d+\.\d+)", content)
    if per:
        return float(per.group(1))
    return None


def collect_per_for_models(exp_dir: Path) -> List[float, str]:
    """
    Collect the PER for all models in the experiment directory
    """
    models = ["mono", "tri1", "tri2b", "tri3b"]
    per_values = []

    for model in models:
        best_wer_file = exp_dir / model / "decode_test/scoring_kaldi/best_wer"
        if os.path.exists(best_wer_file):
            with open(best_wer_file, "r") as f:
                content = f.read()
                match = re.search(r"%WER\s+(\d+\.\d+)", content)
                if match:
                    per = float(match.group(1))
                    per_values.append((model, per))

    return per_values


def plot_per_comparison(
    per_values: List[float, str], output_file: Path
) -> None:
    """
    Plot PER comparison for different models
    """
    # Create a more descriptive mapping for the model names
    model_desc_map = {
        "mono": "Monophone",
        "tri1": "Triphone",
        "tri2b": "Triphone + LDA/MLLT",
        "tri3b": "Triphone + SAT",
    }

    models = [model_desc_map.get(m, m) for m, _ in per_values]
    pers = [p for _, p in per_values]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot the bars with a colormap
    colors = sns.color_palette("Blues_d", len(per_values))
    bars = plt.bar(models, pers, color=colors)

    # Add value labels above bars
    for bar, per in zip(bars, pers):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{per:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title("Phone Error Rate (PER) Comparison", fontsize=16)
    plt.xlabel("Accoustic Model", fontsize=14)
    plt.ylabel("PER (%)", fontsize=14)
    plt.ylim(0, max(pers) * 1.2)  # Add some space above the highest bar

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def analyze_confusion_matrix(exp_dir, model, output_file, top_n=20):
    """
    Analyze phoneme confusion matrix from Kaldi decoding results.

    Args:
        exp_dir (str): Path to Kaldi experiment directory
        model (str): Model name (e.g., 'mono', 'tri1', 'tri2b', 'tri3b')
        output_file (str): Path to save the confusion matrix visualization
        top_n (int): Number of most frequent phonemes to include in visualization

    Returns:
        pd.DataFrame: The full confusion matrix
    """

    # Step 1: Extract reference and hypothesis transcripts from Kaldi output
    decode_dir = os.path.join(exp_dir, model, "decode_test")
    scoring_dir = os.path.join(decode_dir, "scoring_kaldi")

    # Find the best WER hypothesis file
    best_wer_file = os.path.join(scoring_dir, "best_wer")
    if not os.path.exists(best_wer_file):
        print(f"Cannot find best_wer file at {best_wer_file}")
        return None

    # Read the best WER file to get the language model weight
    with open(best_wer_file, "r") as f:
        best_wer_line = f.read().strip()
        # Extract the language model weight from the line
        # Format example: %WER 24.61 [ 3952 / 16059, 673 ins, 768 del, 2511 sub ]
        lmwt = re.search(r"penalty=\d+\.?\d*,lm=(\d+\.?\d*)", best_wer_line)
        if lmwt:
            lmwt = lmwt.group(1)
        else:
            # Default to 11 if not found
            lmwt = "11"

    # Step 2: Get the alignment information
    # Paths to reference and hypothesis files
    ref_file = os.path.join(scoring_dir, f"test_filt.txt")
    hyp_file = os.path.join(scoring_dir, f"penalty_0.0/{lmwt}.txt")

    if not os.path.exists(ref_file) or not os.path.exists(hyp_file):
        print(
            f"Cannot find reference or hypothesis files at {ref_file} or {hyp_file}"
        )
        return None

    # Step 3: Use Kaldi's align-text to get alignments
    align_output_file = os.path.join(scoring_dir, "alignment.txt")

    # Run align-text to get alignment information
    try:
        subprocess.run(
            [
                "align-text",
                "--special-symbol='***'",
                f"ark:{ref_file}",
                f"ark:{hyp_file}",
                f"ark,t:{align_output_file}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running align-text: {e}")
        return None

    # Step 4: Parse the alignment file to build confusion statistics
    confusion_counts = defaultdict(lambda: defaultdict(int))
    insertion_counts = defaultdict(int)
    deletion_counts = defaultdict(int)

    with open(align_output_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            uttid = parts[0]
            alignment = parts[1:]

            i = 0
            while i < len(alignment):
                if alignment[i] == "***":
                    # This is an insertion
                    insertion_counts[alignment[i + 1]] += 1
                    i += 2
                elif i + 1 < len(alignment) and alignment[i + 1] == "***":
                    # This is a deletion
                    deletion_counts[alignment[i]] += 1
                    i += 2
                elif i + 1 < len(alignment):
                    # This is a substitution (could also be correct)
                    ref_phone = alignment[i]
                    hyp_phone = alignment[i + 1]
                    confusion_counts[ref_phone][hyp_phone] += 1
                    i += 2
                else:
                    # Handle any remaining item
                    i += 1

    # Step 5: Create a confusion matrix as pandas DataFrame
    # Get unique phonemes from both reference and hypothesis
    ref_phones = list(confusion_counts.keys())
    hyp_phones = set()
    for ref_phone, hyps in confusion_counts.items():
        hyp_phones.update(hyps.keys())
    all_phones = sorted(set(ref_phones + list(hyp_phones)))

    # Create a DataFrame for the confusion matrix
    confusion_matrix = pd.DataFrame(0, index=all_phones, columns=all_phones)

    # Fill the matrix with counts
    for ref_phone, hyps in confusion_counts.items():
        for hyp_phone, count in hyps.items():
            confusion_matrix.at[ref_phone, hyp_phone] = count

    # Add a column for deletions
    confusion_matrix["<del>"] = 0
    for phone, count in deletion_counts.items():
        if phone in confusion_matrix.index:
            confusion_matrix.at[phone, "<del>"] = count

    # Add a row for insertions
    insertion_row = pd.Series(0, index=confusion_matrix.columns)
    for phone, count in insertion_counts.items():
        if phone in insertion_row.index:
            insertion_row[phone] = count
    confusion_matrix.loc["<ins>"] = insertion_row

    # Step 6: Calculate row-normalized matrix for visualization
    row_sums = confusion_matrix.sum(axis=1)
    normalized_matrix = confusion_matrix.div(row_sums, axis=0).fillna(0)

    # Step 7: Select top phonemes for visualization to avoid an overly large plot
    # Count total occurrences of each phone (rows + columns except special symbols)
    phone_counts = defaultdict(int)
    for phone in all_phones:
        if phone not in ["<del>", "<ins>"]:
            phone_counts[phone] = (
                confusion_matrix.loc[phone].sum()
                + confusion_matrix[phone].sum()
                - confusion_matrix.at[phone, phone]
            )

    # Get top N most frequent phones, plus special symbols
    top_phones = [
        p
        for p, _ in sorted(
            phone_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
    ]
    vis_phones = top_phones + ["<del>", "<ins>"]

    # Create a subset matrix for visualization
    vis_matrix = normalized_matrix.loc[vis_phones, vis_phones]

    # Step 8: Plot the confusion matrix
    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(vis_matrix.values, dtype=bool)
    np.fill_diagonal(
        mask, True
    )  # We'll use a different color for the diagonal

    # Plot the off-diagonal elements
    sns.heatmap(
        vis_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=vis_phones,
        yticklabels=vis_phones,
        mask=mask,
    )

    # Plot the diagonal elements with a different color
    diagonal_values = np.diag(vis_matrix.values)
    for i in range(len(vis_phones)):
        if (
            i < vis_matrix.shape[0] and i < vis_matrix.shape[1]
        ):  # Ensure we're within matrix bounds
            plt.text(
                i + 0.5,
                i + 0.5,
                f"{diagonal_values[i]:.2f}",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="green", alpha=0.6),
            )

    plt.title(f"Phoneme Confusion Matrix - {model.upper()} Model", fontsize=16)
    plt.xlabel("Predicted Phoneme", fontsize=14)
    plt.ylabel("True Phoneme", fontsize=14)

    # Add a textbox with overall stats
    correct = sum(
        confusion_matrix.iloc[i, i]
        for i in range(
            min(confusion_matrix.shape[0] - 1, confusion_matrix.shape[1])
        )
    )
    total = confusion_matrix.iloc[:-1, :].sum().sum()
    accuracy = correct / total if total > 0 else 0

    plt.figtext(
        0.5,
        0.01,
        f"Phoneme Recognition Accuracy: {accuracy:.2f}\n"
        f"Total Substitutions: {total - correct}\n"
        f'Total Insertions: {confusion_matrix.loc["<ins>"].sum()}\n'
        f'Total Deletions: {confusion_matrix["<del>"][:-1].sum()}',
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5),
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=300)
    plt.close()

    # Step 9: Generate additional analysis - find most problematic phonemes
    error_analysis = pd.DataFrame(
        {
            "Correct": np.diag(confusion_matrix.values)[
                :-1
            ],  # Exclude <ins> row
            "Substitutions": (
                confusion_matrix.iloc[:-1].sum(axis=1)
                - np.diag(confusion_matrix.values)[:-1]
            ),
            "Deletions": confusion_matrix["<del>"][:-1],
            "Total": confusion_matrix.iloc[:-1].sum(axis=1),
        }
    )
    error_analysis["Accuracy"] = (
        error_analysis["Correct"] / error_analysis["Total"]
    )

    # Save error analysis to a separate file
    error_file = output_file.replace(".png", "_error_analysis.csv")
    error_analysis.sort_values(by="Accuracy").to_csv(error_file)

    # Also generate a phoneme error distribution plot
    error_plot_file = output_file.replace(".png", "_error_distribution.png")
    top_error_phones = (
        error_analysis.sort_values(by="Accuracy").head(15).index.tolist()
    )

    plt.figure(figsize=(12, 8))
    error_data = error_analysis.loc[top_error_phones].sort_values(
        by="Accuracy"
    )

    # Create a stacked bar chart
    bar_width = 0.8
    error_data[["Correct", "Substitutions", "Deletions"]].plot(
        kind="bar",
        stacked=True,
        color=["green", "orange", "red"],
        figsize=(12, 8),
    )

    # Add accuracy as a line on secondary axis
    ax2 = plt.twinx()
    ax2.plot(
        range(len(top_error_phones)),
        error_data["Accuracy"],
        "o-",
        color="blue",
        linewidth=2,
    )
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Accuracy")

    plt.title(
        "Phoneme Error Distribution for Most Problematic Phonemes", fontsize=16
    )
    plt.xlabel("Phoneme", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(
        ["Accuracy", "Correct", "Substitutions", "Deletions"], loc="best"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(error_plot_file, dpi=300)
    plt.close()

    print(
        f"Confusion matrix and error analysis saved to {output_file} and {error_file}"
    )
    return confusion_matrix
