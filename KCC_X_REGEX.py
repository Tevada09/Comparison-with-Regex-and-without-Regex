"""
Test script for khmer-nlp-kcc 0.2.8 with Regex
Reads Khmer text from dataset.csv, preprocesses text using Regex,
runs segmentation, POS tagging, and sentiment analysis,
then writes all results to kcc_regx_result.csv.
"""

import csv
import json
import re
from khmer_nlp import KhmerNLP

INPUT_FILE = "input.csv"
OUTPUT_FILE = "kcc_regx_result_1.csv"


# ==========================================
# 1. REGEX TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    # Convert input to string and remove surrounding spaces
    text = str(text).strip()

    # Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URL ',
        text,
        flags=re.IGNORECASE
    )

    # Replace email addresses
    text = re.sub(
        r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b',
        ' EMAIL ',
        text
    )

    # Normalize currency and percentage symbols
    text = re.sub(r'\$', ' ដុល្លារ ', text)
    text = re.sub(r'៛', ' រៀល ', text)
    text = re.sub(r'%', ' ភាគរយ ', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    # Replace repeated whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================================
# 2. MAIN NLP PROGRAM
# ==========================================

def main():
    print("Loading KhmerNLP model ...")
    nlp = KhmerNLP()

    rows = []

    # Read input CSV
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        inputs = list(reader)

    total = len(inputs)
    print(f"Processing {total} sentence(s) ...\n")

    for i, row in enumerate(inputs, 1):

        # Keep original text unchanged
        original_text = row["text"].strip()

        # Apply Regex preprocessing
        cleaned_text = preprocess_text(original_text)

        print(f"[{i}/{total}]")
        print(f"Original : {original_text}")
        print(f"Processed: {cleaned_text}")

        # Word segmentation
        segmented = nlp.segment(cleaned_text)

        # Standard POS tagging
        pos_tags = nlp.pos(cleaned_text)
        pos_str = " | ".join(
            f'{d["word"]}({d["label"]})'
            for d in pos_tags
        )

        # Nova POS tagging
        nova_tags = nlp.nova_pos(cleaned_text)
        nova_str = " | ".join(
            f'{d["word"]}({d["label"]})'
            for d in nova_tags
        )

        # Sentiment analysis
        polarity = nlp.polarity(cleaned_text)

        pol_label = polarity["label"]
        pol_conf = polarity["confidence"]
        pol_all = json.dumps(
            polarity["scores"],
            ensure_ascii=False
        )

        # Store results
        rows.append({
            "id": i,
            "original_text": original_text,
            "processed_text": cleaned_text,
            "segmented": segmented,
            "pos_tags": pos_str,
            "nova_pos": nova_str,
            "polarity": pol_label,
            "confidence": pol_conf,
            "all_scores": pol_all,
        })

        print(f"Segmented: {segmented}")
        print(f"POS tags : {pos_str}")
        print(f"Sentiment: {pol_label} ({pol_conf})\n")

    # Write results to CSV
    fieldnames = [
        "id",
        "original_text",
        "processed_text",
        "segmented",
        "pos_tags",
        "nova_pos",
        "polarity",
        "confidence",
        "all_scores",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Results saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()