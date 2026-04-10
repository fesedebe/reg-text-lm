# Legal Text Simplifier

Fine-tunes open-source 7B models (Qwen 2.5-7B-Instruct, Llama 3.1-8B-Instruct) using QLoRA to rewrite legal text in plain English, while preserving legal meaning. Fine-tuned Qwen 2.5-7B matches or beats frontier API models (GPT-4o-mini, GPT-5.4) on this task, running locally at zero per-query cost.

## Quick Start

```bash
pip install -r requirements.txt

# Training
python src/train.py --model qwen
python src/merge.py --model qwen

# Serving & Inference
VLLM_API_KEY="your-key" python src/serve.py --model qwen
python src/inference.py --model qwen
```

Call the API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")

response = client.chat.completions.create(
    model="legal-simplifier",
    messages=[{"role": "user", "content": "Simplify: The defendant shall comply..."}]
)
print(response.choices[0].message.content)
```
## Project Structure

```
src/        # Core modules (training, inference, serving, evaluation)
scripts/    # Data prep, external API prediction calls
tests/      # pytest tests
docs/       # Evals & methods report
data/       # Training & test data
output/     # Model checkpoints & predictions
```
## Docker

```bash
docker build -f docker/Dockerfile -t legal-simplification .
docker run --gpus all -it legal-simplification
```

## Evaluation

- **Results and analysis:** [docs/eval_report.md](docs/eval_report.md)
- **Methodology and setup:** [docs/methods.md](docs/methods.md)

## Example

**Input** (grade level 20.1):
> The applicant finally complained of an impairment of the principle of equality of arms, as its appeal on points of law had been declared inadmissible by the Supreme Court, while the appeal based on the same ground submitted by the defendant had been granted.

**Output** (grade level 9.5):
> Finally, the applicant complained about the infringement of the principle of equal treatment. The applicant's appeal on legal grounds was rejected by the Supreme Court. However, the defendant's appeal on the same grounds was accepted.