#eval.py
#Evaluation to compare models' predictions.

import json
from difflib import SequenceMatcher
from pathlib import Path
import textstat

def load_predictions(input_dir: str, files: list[str] | None = None) -> dict[str, list[dict]]:
    #Load .jsonl prediction files from directory, optionally filtered by stem
    contestants = {}
    for f in sorted(Path(input_dir).glob("*.jsonl")):
        key = f.stem
        if files is not None and key not in files:
            continue
        with open(f) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        contestants[key] = records
        print(f"Loaded {key}: {len(records)} examples")
    return contestants

def compute_metrics(records: list[dict]) -> dict:
    #Compute per-example and aggregate metrics for one contestant
    readability_deltas = []
    change_rates = []
    length_ratios = []
    input_fkgls = []
    output_fkgls = []

    for r in records:
        inp = r["input"]
        out = r["model_output"]

        fk_in = textstat.flesch_kincaid_grade(inp)
        fk_out = textstat.flesch_kincaid_grade(out)
        readability_deltas.append(fk_out - fk_in)
        input_fkgls.append(fk_in)
        output_fkgls.append(fk_out)

        similarity = SequenceMatcher(None, inp, out).ratio()
        change_rates.append(1 - similarity)

        length_ratios.append(len(out) / len(inp) if len(inp) > 0 else 1.0)

    n = len(records)
    return {
        "readability_deltas": readability_deltas,
        "change_rates": change_rates,
        "length_ratios": length_ratios,
        "input_fkgls": input_fkgls,
        "output_fkgls": output_fkgls,
        "avg_readability_delta": sum(readability_deltas) / n,
        "avg_change_rate": sum(change_rates) / n,
        "avg_length_ratio": sum(length_ratios) / n,
        "avg_input_fkgl": sum(input_fkgls) / n,
        "avg_output_fkgl": sum(output_fkgls) / n,
    }

def display_name(key: str, names: dict[str, str] | None = None) -> str:
    #Return display name for a key, or the key itself if no mapping
    if names is None:
        return key
    return names.get(key, key)

def sort_contestants(keys: list[str], order: list[str] | None = None) -> list[str]:
    #Sort contestant keys by given order, unknowns at end. Alphabetical if no order.
    if order is None:
        return sorted(keys)
    order_map = {k: i for i, k in enumerate(order)}
    return sorted(keys, key=lambda k: order_map.get(k, 999))
