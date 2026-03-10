#combine_predictions.py
#Combines prediction JSONLs into browseable CSV + JSONL with per-example metrics.
#
#Usage: python scripts/combine_predictions.py

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.eval import load_predictions, compute_metrics, display_name, sort_contestants

INPUT_DIR = "output/comparison"

DISPLAY_NAMES = {
    "base-qwen": "Base Qwen 2.5-7B",
    "base-llama": "Base Llama 3.1-8B",
    "ft-qwen": "Fine-tuned Qwen 2.5-7B",
    "ft-llama": "Fine-tuned Llama 3.1-8B",
    "fewshot-gpt-4o-mini": "Few-shot GPT-4o-mini",
    "zeroshot-gpt-4o-mini": "Zero-shot GPT-4o-mini",
    "fewshot-gpt-5.4": "Few-shot GPT-5.4",
    "zeroshot-gpt-5.4": "Zero-shot GPT-5.4",
}

DISPLAY_ORDER = [
    "base-qwen", "base-llama", "ft-qwen", "ft-llama",
    "fewshot-gpt-4o-mini", "zeroshot-gpt-4o-mini",
    "fewshot-gpt-5.4", "zeroshot-gpt-5.4",
]

def main():
    parser = argparse.ArgumentParser(
        description="Combine prediction files into browseable CSV + JSONL with metrics"
    )
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input-dir)")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Prediction file stems to include (e.g. base-qwen ft-qwen)")
    parser.add_argument("--include-gold", action="store_true",
                        help="Include gold_output column")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    contestants = load_predictions(args.input_dir, files=args.files)
    contestants.pop("combined", None)
    if not contestants:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    # Compute metrics per contestant
    all_metrics = {}
    for key, records in contestants.items():
        all_metrics[key] = compute_metrics(records)

    keys = sort_contestants(list(contestants.keys()), DISPLAY_ORDER)
    n_examples = len(next(iter(contestants.values())))

    # Build combined records
    combined = []
    for i in range(n_examples):
        ref_record = next(iter(contestants.values()))[i]
        entry = {
            "example": i + 1,
            "input": ref_record["input"],
        }
        if args.include_gold:
            entry["gold_output"] = ref_record.get("gold_output", "")

        entry["contestants"] = {}
        for k in keys:
            record = contestants[k][i]
            entry["contestants"][k] = {
                "output": record["model_output"],
                "readability_delta": round(all_metrics[k]["readability_deltas"][i], 2),
                "change_rate": round(all_metrics[k]["change_rates"][i], 4),
                "length_ratio": round(all_metrics[k]["length_ratios"][i], 2),
                "input_fkgl": round(all_metrics[k]["input_fkgls"][i], 2),
                "output_fkgl": round(all_metrics[k]["output_fkgls"][i], 2),
            }
        combined.append(entry)

    os.makedirs(output_dir, exist_ok=True)

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "combined.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in combined:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {jsonl_path}")

    # Write CSV
    csv_path = os.path.join(output_dir, "combined.csv")
    fieldnames = ["example", "input"]
    if args.include_gold:
        fieldnames.append("gold_output")
    for k in keys:
        name = display_name(k, DISPLAY_NAMES)
        fieldnames.extend([
            f"{name} — output",
            f"{name} — delta",
            f"{name} — change_rate",
            f"{name} — length_ratio",
            f"{name} — input_fkgl",
            f"{name} — output_fkgl",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in combined:
            row = {
                "example": entry["example"],
                "input": entry["input"],
            }
            if args.include_gold:
                row["gold_output"] = entry.get("gold_output", "")
            for k in keys:
                name = display_name(k, DISPLAY_NAMES)
                c = entry["contestants"][k]
                row[f"{name} — output"] = c["output"]
                row[f"{name} — delta"] = c["readability_delta"]
                row[f"{name} — change_rate"] = f"{c['change_rate']:.2%}"
                row[f"{name} — length_ratio"] = c["length_ratio"]
                row[f"{name} — input_fkgl"] = c["input_fkgl"]
                row[f"{name} — output_fkgl"] = c["output_fkgl"]
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    print(f"\n{n_examples} examples x {len(keys)} contestants")

if __name__ == "__main__":
    main()
