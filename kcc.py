"""
Test script for khmer-nlp-kcc 0.2.8
Reads Khmer text from input.csv, runs segmentation, POS tagging,
and sentiment polarity, then writes all results to result_1.csv.
"""

import csv
import json
from khmer_nlp import KhmerNLP

INPUT_FILE  = "input_2.csv"
OUTPUT_FILE = "result_2.csv"

def main():
    # Initialize the model (downloads checkpoint from HuggingFace on first run)
    print("Loading KhmerNLP model ...")
    nlp = KhmerNLP()

    rows = []

    # Read input
    with open(INPUT_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        inputs = list(reader)

    total = len(inputs)
    print(f"Processing {total} sentence(s) ...\n")

    for i, row in enumerate(inputs, 1):
        text = row["text"].strip()
        print(f"[{i}/{total}] {text}")
        row["id"] = i  # auto-generate id

        # --- Word segmentation ---
        segmented = nlp.segment(text)

        # --- POS tagging (returns list of {"word": ..., "label": ...}) ---
        pos_tags = nlp.pos(text)
        pos_str  = " | ".join(f'{d["word"]}({d["label"]})' for d in pos_tags)

        # --- Nova POS (extended tag set) ---
        nova_tags = nlp.nova_pos(text)
        nova_str  = " | ".join(f'{d["word"]}({d["label"]})' for d in nova_tags)

        # --- Sentiment polarity ---
        polarity  = nlp.polarity(text)
        pol_label = polarity["label"]
        pol_conf  = polarity["confidence"]
        pol_all   = json.dumps(polarity["scores"], ensure_ascii=False)

        rows.append({
            "id":           row["id"],
            "original_text": text,
            "segmented":    segmented,
            "pos_tags":     pos_str,
            "nova_pos":     nova_str,
            "polarity":     pol_label,
            "confidence":   pol_conf,
            "all_scores":   pol_all,
        })

        print(f"  segmented : {segmented}")
        print(f"  pos_tags  : {pos_str}")
        print(f"  polarity  : {pol_label} ({pol_conf})\n")

    # Write results
    fieldnames = [
        "id", "original_text", "segmented",
        "pos_tags", "nova_pos",
        "polarity", "confidence", "all_scores",
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Results saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
