#run_external_api.py
#Few-shot and zero-shot inference via external APIs (OpenAI, Anthropic); baseline evals to compare with fine-tuned models.
#
#Usage: python scripts/run_external_api.py --provider openai --model gpt-4o-mini --shots 5

import argparse
import json
import os
import re

# Canonical version from src/config.py DataConfig.system_prompt
SYSTEM_PROMPT = (
    "You rewrite legal text in plain English that non-lawyers can understand. "
    "Preserve ALL information, dates, names, citations, conditions, and exceptions. "
    "Use simpler words but keep the full legal meaning intact. "
    "Do not soften requirements, omit edge cases, or add explanations."
)

TEST_DATA = "data/processed/qlora_test_filtered.jsonl"
DEFAULT_OUTPUT_DIR = "output/comparison"

# 5 few-shot examples from training data
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            'On 29 October 2001 the prosecutor refused to institute criminal '
            'proceedings because of the lack of evidence that a crime had been '
            'committed. In particular, he found that the applicant had attempted '
            'to flee and had resisted the policemen.'
        ),
        "output": (
            'On October 29, 2001, the prosecutor refused to start criminal '
            'proceedings because of the lack of evidence that a crime had been '
            'committed. In particular, he found that the applicant had attempted '
            'to escape and had resisted the policemen.'
        ),
    },
    {
        "input": (
            'By an additional judgment dated 4 May 2004, the Ankara State Security '
            'Court, taking into account the amendment to the definition of "terror '
            'acts" under section 7 of Law no. 3713 pursuant to Law no. 4928, '
            'acquitted the applicant, who had not taken part in any violent '
            'activities, and nullified his conviction, together with all its '
            'consequences. This judgment became final on 11 May 2004.'
        ),
        "output": (
            'By an additional decision dated 4 May 2004, the Ankara State Security '
            'Court found the applicant not guilty. This decision took into account '
            'the amendment to the definition of "terror acts" under section 7 of '
            'Law no. 3713 pursuant to Law no. 4928. The applicant, who had not '
            'taken part in any violent activities, had his conviction canceled, '
            'along with all its consequences. This decision was finalized on '
            '11 May 2004.'
        ),
    },
    {
        "input": (
            'On 30 June 1997 the applicant lodged a request for restitution of '
            'the confiscated property with an administrative body, which has not '
            'yet issued a decision.'
        ),
        "output": (
            'On June 30, 1997, the applicant asked an administrative agency to '
            'return the property. The agency has not yet decided.'
        ),
    },
    {
        "input": (
            'While a federal court may exercise supplemental jurisdiction over '
            'state law claims "that are so related to claims in the action within '
            '[the Court\'s] original jurisdiction that they form a part of the same '
            'case or controversy under Article III of the United States '
            'Constitution," a court may decline to exercise such jurisdiction '
            'where it "has dismissed all claims over which it has original '
            'jurisdiction." Indeed, unless "consideration of judicial economy, '
            'convenience and fairness to litigants" weigh in favor of the exercise '
            'of supplemental jurisdiction, "a federal court should hesitate to '
            'exercise jurisdiction over state claims." Because Plaintiffs complaint '
            'fails to state a viable federal claim, and because this case is at '
            'the beginning stages of litigation, the Court declines to exercise '
            'supplemental jurisdiction over Plaintiffs state law claims at this time.'
        ),
        "output": (
            "A federal court may give extra decisions over state law if the state "
            "law claims are closely related to the main claims in the case. "
            "However, if the court dismisses all the primary claims, it can choose "
            "not to handle the state law claims. If it's not more efficient, "
            "convenient, and fair for everyone involved, the court should be "
            "cautious about handling state law claims. Because the Plaintiffs' "
            "complaint doesn't have a valid federal claim, and because this case "
            "is in the early stages of the legal process, the Court declines to "
            "give extra decisions over the Plaintiffs' state law claims right now."
        ),
    },
    {
        "input": (
            'In this matter, Dr. Pence offers four opinions: (1) BSC did not '
            'conduct adequate testing of the Pinnacle product prior to placing '
            'them on the market; (2) the Pinnacle product was inadequately labeled; '
            '(3) patients could not adequately consent to the surgical implantation '
            'of the Pinnacle due to the misbranding of these products; and (4) BSC '
            'failed to meet the post-market vigilance standard of care for their '
            'products, leading to further mis-branding.'
        ),
        "output": (
            "In this case, Dr. Pence has four opinions: (1) BSC didn't test the "
            "Pinnacle product enough before selling it; (2) the Pinnacle product "
            "didn't have proper labels; (3) patients couldn't give proper consent "
            "for the surgical implantation of the Pinnacle because of the incorrect "
            "labeling of these products; and (4) BSC didn't meet the post-market "
            "safety standards for their products, which caused more incorrect "
            "branding."
        ),
    },
]

INSTRUCTION = "Rewrite this in plain English."

def build_messages(test_input: str, shots: int = 5) -> list:
    #Build message list with system prompt, optional few-shot examples, and test input
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if shots > 0:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": f"{INSTRUCTION}\n\n{ex['input']}"})
            messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": f"{INSTRUCTION}\n\n{test_input}"})
    return messages

def call_openai(messages: list, model: str, base_url: str | None,
                temperature: float, max_tokens: int) -> str:
    #Call OpenAI-compatible API
    from openai import OpenAI

    kwargs = {"api_key": os.environ.get("OPENAI_API_KEY")}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # Newer OpenAI models (e.g. gpt-5.4) require max_completion_tokens
    if "gpt-5" in model or "o1" in model or "o3" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        completion_kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**completion_kwargs)
    return response.choices[0].message.content.strip()

def call_anthropic(messages: list, model: str, temperature: float,
                   max_tokens: int) -> str:
    #Call Anthropic API
    import anthropic

    client = anthropic.Anthropic()

    system = messages[0]["content"]
    chat_messages = messages[1:]

    response = client.messages.create(
        model=model,
        system=system,
        messages=chat_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content[0].text.strip()

def sanitize_filename(name: str) -> str:
    #Turn model name into a safe filename
    return re.sub(r"[^a-zA-Z0-9_\-.]", "-", name)

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run few-shot or zero-shot predictions via API for comparison study"
    )
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    parser.add_argument("--model", required=True, help="Model name as the API expects it")
    parser.add_argument("--shots", type=int, choices=[0, 5], default=5,
                        help="Number of few-shot examples (0 for zero-shot, 5 for few-shot)")
    parser.add_argument("--base-url", default=None, help="Override API base URL (openai provider only)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None,
                        help="Output filename without extension (default: {zeroshot|fewshot}-{model})")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.output_name:
        output_name = args.output_name
    else:
        prefix = "zeroshot" if args.shots == 0 else "fewshot"
        output_name = f"{prefix}-{sanitize_filename(args.model)}"
    output_path = os.path.join(args.output_dir, f"{output_name}.jsonl")

    with open(TEST_DATA) as f:
        test_examples = [json.loads(line) for line in f]

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Shots: {args.shots}")
    print(f"Temperature: {args.temperature}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Output: {output_path}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for i, ex in enumerate(test_examples):
        original = ex.get("input", "").strip()
        gold = ex.get("output", "").strip()

        messages = build_messages(original, shots=args.shots)

        try:
            if args.provider == "openai":
                model_output = call_openai(
                    messages, args.model, args.base_url,
                    args.temperature, args.max_tokens,
                )
            else:
                model_output = call_anthropic(
                    messages, args.model,
                    args.temperature, args.max_tokens,
                )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            model_output = f"ERROR: {e}"

        record = {
            "instruction": INSTRUCTION,
            "input": original,
            "gold_output": gold,
            "model_output": model_output,
        }
        records.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_examples):
            print(f"Processed {i+1}/{len(test_examples)} examples")

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    errors = sum(1 for r in records if r["model_output"].startswith("ERROR:"))
    print(f"\nDone. Saved {len(records)} predictions to {output_path}")
    if errors:
        print(f"  ({errors} failed)")

if __name__ == "__main__":
    main()

#Usage:
#  python scripts/run_external_api.py --provider openai --model gpt-4o-mini --shots 5
#  python scripts/run_external_api.py --provider openai --model gpt-5.4 --shots 0
#  python scripts/run_external_api.py --provider anthropic --model claude-sonnet-4-6
#  python scripts/run_external_api.py --provider openai --model deepseek-chat --base-url https://api.deepseek.com/v1