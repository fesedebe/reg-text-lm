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

# Excellent test examples to move to training
EXCELLENT_TEST_LINES = [11, 31, 33, 40, 41, 43, 48]

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
    if similarity > 0.935:
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

def load_jsonl(path: str) -> List[Dict]:
    #Load JSONL file into list of dicts.
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def save_jsonl(examples: List[Dict], path: str):
    #Save list of dicts to JSONL file.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def filter_and_curate(
    train_input: str,
    test_input: str,
    train_output: str,
    test_output: str,
    holdout_output: str,
) -> Dict:
    #Filter train/test data and curate splits.
    
    # Load data
    train_examples = load_jsonl(train_input)
    test_examples = load_jsonl(test_input)
    
    print(f"Loaded {len(train_examples)} train examples from {train_input}")
    print(f"Loaded {len(test_examples)} test examples from {test_input}")
    
    # Process train data
    train_kept = []
    holdout = []
    train_removal_reasons = {}
    
    for example in train_examples:
        keep, reason, fixed_example = filter_example(example)
        
        if keep:
            train_kept.append(fixed_example)
        else:
            holdout.append(example) 
            train_removal_reasons[reason] = train_removal_reasons.get(reason, 0) + 1
    
    # Process test data
    test_kept = []
    excellent_for_train = []
    test_removal_reasons = {}
    
    for i, example in enumerate(test_examples, start=1):
        # Check if this is an excellent example to move to train
        if i in EXCELLENT_TEST_LINES:
            excellent_for_train.append(example)
            continue
        
        # Apply same filtering as train
        keep, reason, fixed_example = filter_example(example)
        
        if keep:
            test_kept.append(fixed_example)
        else:
            holdout.append(example)
            test_removal_reasons[reason] = test_removal_reasons.get(reason, 0) + 1
    
    # Add excellent test examples to train
    train_final = train_kept + excellent_for_train
    
    # Save outputs
    save_jsonl(train_final, train_output)
    save_jsonl(test_kept, test_output)
    save_jsonl(holdout, holdout_output)
    
    # Print summary
    print("FILTERING COMPLETE")
    print(f"\nTRAIN:")
    print(f"  Original: {len(train_examples)}")
    print(f"  Kept: {len(train_kept)}")
    print(f"  + Excellent from test: {len(excellent_for_train)}")
    print(f"  Final: {len(train_final)}")
    print(f"  Removed: {len(train_examples) - len(train_kept)}")
    if train_removal_reasons:
        print(f"  Removal breakdown:")
        for reason, count in sorted(train_removal_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    print(f"\nTEST:")
    print(f"  Original: {len(test_examples)}")
    print(f"  Moved to train (excellent): {len(excellent_for_train)}")
    print(f"  Kept for test: {len(test_kept)}")
    print(f"  Moved to holdout: {len(test_examples) - len(excellent_for_train) - len(test_kept)}")
    if test_removal_reasons:
        print(f"  Removal breakdown:")
        for reason, count in sorted(test_removal_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    print(f"\nHOLDOUT (for manual review):")
    print(f"  Total: {len(holdout)}")
    
    print(f"\nSaved files:")
    print(f"  {train_output} ({len(train_final)} examples)")
    print(f"  {test_output} ({len(test_kept)} examples)")
    print(f"  {holdout_output} ({len(holdout)} examples)")
    
    return {
        "train_original": len(train_examples),
        "train_final": len(train_final),
        "test_original": len(test_examples),
        "test_final": len(test_kept),
        "holdout": len(holdout),
        "excellent_moved": len(excellent_for_train),
    }

def main():
    stats = filter_and_curate(
        train_input="data/processed/qlora_train.jsonl",
        test_input="data/processed/qlora_test.jsonl",
        train_output="data/processed/qlora_train_filtered.jsonl",
        test_output="data/processed/qlora_test_filtered.jsonl",
        holdout_output="data/processed/qlora_holdout.jsonl",
    )
    
    return stats

if __name__ == "__main__":
    main()
