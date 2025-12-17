# Filters dataset to remove problematic examples

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

# Wrong terms
BAD_TERMS = [
    "suers",          
    "subpeana",       
    "younglings",     
    "stinging allegations",  
]

# Typo corrections
TYPO_FIXES = {
    "lenghty": "lengthy",
    "diffucult": "difficult",
    "rejecet": "rejected",
    "tthe": "the",
}

def calculate_similarity(text1: str, text2: str) -> float:
    #Calculate similarity ratio between two texts.
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def contains_bad_terms(text: str) -> bool:
    #Check if text contains any known bad terms.
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in BAD_TERMS)

def fix_typos(text: str) -> str:
    #Fix common typos in text.
    for typo, correction in TYPO_FIXES.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(typo), re.IGNORECASE)
        text = pattern.sub(correction, text)
    return text

def is_truncated(input_text: str, output_text: str, threshold: float = 0.3) -> bool:
    #Check if output is significantly shorter than input.
    if len(input_text) == 0:
        return False
    ratio = len(output_text) / len(input_text)
    return ratio < (1 - threshold)

def filter_example(example: Dict) -> Tuple[bool, str, Dict]:
    #Filter a single example.
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    instruction = example.get("instruction", "").strip()
    
    if not input_text or not output_text:
        return False, "empty_field", example
    
    if input_text == output_text:
        return False, "identical", example
    
    similarity = calculate_similarity(input_text, output_text)
    if similarity > 0.95:
        return False, f"near_identical_{similarity:.2f}", example
    
    if contains_bad_terms(output_text):
        return False, "bad_terms", example
    
    if is_truncated(input_text, output_text, threshold=0.4):
        return False, "truncated", example
    
    fixed_output = fix_typos(output_text)
    
    fixed_example = {
        "instruction": instruction,
        "input": input_text,
        "output": fixed_output,
    }
    
    return True, "kept", fixed_example

def filter_dataset(input_path: str, output_path: str) -> Dict:
    #Filter the dataset and save to output path.
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples from {input_path}")
    
    kept_examples = []
    removal_reasons = {}
    
    for example in examples:
        keep, reason, fixed_example = filter_example(example)
        
        if keep:
            kept_examples.append(fixed_example)
        else:
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
    
    # Save filtered examples
    with open(output_path, "w", encoding="utf-8") as f:
        for example in kept_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"\nFiltering complete!")
    print(f"  Original: {len(examples)} examples")
    print(f"  Kept: {len(kept_examples)} examples")
    print(f"  Removed: {len(examples) - len(kept_examples)} examples")
    print(f"\nRemoval breakdown:")
    for reason, count in sorted(removal_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print(f"\nSaved filtered dataset to: {output_path}")
    
    return {
        "original_count": len(examples),
        "kept_count": len(kept_examples),
        "removed_count": len(examples) - len(kept_examples),
        "removal_reasons": removal_reasons,
    }

def main():
    input_path = "data/processed/qlora_train.jsonl"
    output_path = "data/processed/qlora_train_filtered.jsonl"
    
    stats = filter_dataset(input_path, output_path)
    
    return stats

if __name__ == "__main__":
    main()
