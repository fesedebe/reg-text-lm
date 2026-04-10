# Evaluating specialized small and general large language models on text simplification

Can a 7B model fine-tuned on 350 examples outperform frontier API models at making legal text readable?

---

## Summary

Fine-tuned Qwen 2.5-7B is the strongest model in this evaluation. It produces the lowest adjusted output grade level (11.6 FKGL), wins a plurality of head-to-head comparisons against every API baseline tested, and runs on local hardware with zero per-query cost. The value of fine-tuning depends on the base model: it stabilized Llama (eliminating catastrophic FKGL blowups) and taught Qwen to make more targeted edits.

For full methodology details, see [methods.md](methods.md).

## 1. Aggregate Results

Input texts average grade level 14.7 (college-level reading). All models reduce this, but by varying degrees.


| Model                 | Output FKGL (↓) | Adj. FKGL | Delta     | Change Rate | Length Ratio | Success   | Near-copy |
| --------------------- | --------------- | --------- | --------- | ----------- | ------------ | --------- | --------- |
| Zero-shot GPT-4o-mini | 11.5            | 11.5      | −3.17     | 55.8%       | 0.95         | 21/30     | 1/30      |
| **FT Qwen 2.5-7B**    | **11.5**        | **11.6**  | **−3.25** | **45.0%**   | **0.97**     | **21/30** | **3/30**  |
| Base Qwen 2.5-7B      | 11.6            | 11.6      | −3.16     | 59.9%       | 0.92         | 21/30     | 0/30      |
| FT Llama 3.1-8B       | 11.5            | 11.9      | −3.20     | 29.0%       | 0.99         | 13/30     | 10/30     |
| Few-shot GPT-4o-mini  | 12.0            | 12.0      | −2.75     | 52.6%       | 0.95         | 24/30     | 0/30      |
| Few-shot GPT-5.4      | 12.0            | 12.0      | −2.74     | 39.8%       | 0.98         | 21/30     | 4/30      |
| Zero-shot GPT-5.4     | 12.8            | 12.8      | −1.89     | 46.0%       | 1.00         | 19/30     | 4/30      |
| Base Llama 3.1-8B     | 13.6            | 13.6      | −1.16     | 71.6%       | 1.06         | 16/30     | 0/30      |


**Reading the table:** 

- Output FKGL is the absolute grade level of the simplified text (lower = more readable). 
- Delta is the change from input FKGL (negative = simpler). 
- Adjusted FKGL (using a 10% change-rate threshold) replaces output FKGL with input FKGL on near-copies, giving no credit for unchanged text. 
- Success counts examples where change rate > 10%, length ratio is 0.75–1.15, and output FKGL < input FKGL.

The top four models cluster between 11.5 and 11.9 adjusted FKGL. At n=30, margins between them are directional i.e. the behavioral differences matter more than the aggregate scores. In general, aggregate scores obscure very different behavior patterns. 

## 2. Q1: Does fine-tuning help?

**For Qwen: barely on aggregate, but meaningfully in behavior.** Fine-tuned Qwen and base Qwen split 15–15 in head-to-head output FKGL, with a mean improvement of just 0.1 grade levels. But the behavior changed. Fine-tuned Qwen rewrites less (45% change rate vs. 60%) while achieving the same readability, basically making more surgical edits. It also controls length almost perfectly (mean ratio 0.97, stdev 0.08) compared to the base model's occasional over-compression (3 examples below 0.75 length ratio). Fine-tuning taught Qwen to be more precise rather than more aggressive.

**For Llama: big reliability gains, but a new failure mode.** Base Llama was the worst performer overall (13.6 mean output FKGL) with extreme variance (stdev 7.3, nearly double any other model). Fine-tuning cut the mean output FKGL by 2.0 grade levels and collapsed the variance to 3.9, in line with the best models. The cost was a near-copy problem where 10 of 30 examples came back essentially unchanged, including 5 that were byte-for-byte identical to the input. This is the worst failure mode in the evaluation.

### What this looks like in practice

This example, a habeas corpus citation, shows both the near-copy problem and what good simplification looks like:

> **Input** (FKGL 11.7): *Where, as here, a prisoner "challenges the very fact or duration of his physical imprisonment, and the relief he seeks is a determination that he is entitled to immediate release or a speedier release from that imprisonment, his sole federal remedy is a writ of habeas corpus."*

> **FT Llama** (FKGL 11.7, change 0.0%): Returned the input verbatim. No simplification attempted.

> **FT Qwen** (FKGL 9.8, change 63%): *If a prisoner challenges the fact or length of their physical confinement, and the remedy they seek is immediate or faster release, the only federal remedy available is a writ of habeas corpus.*

Fine-tuned Qwen broke the phrasing into clearer structure, swapped "duration" for "length" and "speedier" for "faster," and dropped the nested quotation, all while preserving every legal element. Fine-tuned Llama didn't try. Notably, base Llama changed 88.2% of this example.

## 3. Q2: Can fine-tuned small models beat API models?

Domain-specific fine-tuning on 350 examples is enough to match or exceed general-purpose API models on this task, at zero per-query cost. Fine-tuned Qwen 2.5-7B wins a majority of head-to-head comparisons against every API baseline:


| Comparison               | FT Qwen wins | API wins |
| ------------------------ | ------------ | -------- |
| vs Zero-shot GPT-4o-mini | 17           | 13       |
| vs Few-shot GPT-4o-mini  | 16           | 13       |
| vs Zero-shot GPT-5.4     | 21           | 9        |
| vs Few-shot GPT-5.4      | 17           | 13       |


The 21–9 margin against zero-shot GPT-5.4 is the widest in the evaluation.

### GPT-5.4's performance

GPT-5.4 didn't provide additional value over the cheaper GPT-4o-mini on any metric tested. It produced higher output FKGL than GPT-4o-mini in zero-shot (12.8 vs 11.5) and matched it in few-shot (12.0 vs 12.0). It also had 4 near-copies in both variants. Its near-copies were on already simple inputs (FKGL 6.3–9.7), but the task instruction says "rewrite in plain English," and unchanged text doesn't satisfy it.

When GPT-5.4 did rewrite, it tended to make word-level substitutions while preserving complex sentence structures. The difference can be seen in this example (a 56-word legal sentence, FKGL 22.9):

> **Input** (FKGL 22.9): *A licence must be revoked if it has been misused in such a way that the holder can no longer be deemed suitable to provide the service, though in less serious cases a warning may be issued; it must also be revoked if the service is not maintained (Chapter 3, sections 1 and 2, of the 1979 Act).*

> **FT Qwen** (FKGL 9.4, change 64%): *A license must be canceled if it was used improperly to the point where the holder can no longer be considered appropriate to provide the service. However, in less severe cases, a warning may be given. The license must also be canceled if the service is not maintained (Article 3, sections 1 and 2, of the 1979 Act).*

> **Zero-shot GPT-5.4** (FKGL 13.6, change 66%): *A licence must be taken away if it has been used improperly in a way that means the holder can no longer be considered suitable to provide the service, although in less serious cases a warning may be given instead. It must also be taken away if the service is not maintained (Chapter 3, sections 1 and 2, of the 1979 Act).*

Both models rewrote roughly the same amount of text, but FT Qwen split the run-on into three sentences while GPT-5.4 preserved the semicolon structure and subordinate clauses. This pattern repeats across the evaluation. In this example (FKGL 20.1), the gap was even wider:

> **FT Qwen** (FKGL 9.5, change 83%): *Finally, the applicant complained about the infringement of the principle of equal treatment. The applicant's appeal on legal grounds was rejected by the Supreme Court. However, the defendant's appeal on the same grounds was accepted.*

> **Zero-shot GPT-5.4** (FKGL 17.8, change 81%): *Lastly, the applicant complained that the principle of equality of arms had been undermined because the Supreme Court declared its appeal on points of law inadmissible, while granting the defendant's appeal based on the same ground.*

GPT-5.4 kept the single-sentence structure with a "because...while" construction. FT Qwen broke it into three declarative sentences, which is an 8.3 grade-level gap between the two outputs.

### GPT-4o-mini: the most reliable API model

FT Qwen wins on head-to-head FKGL, but GPT-4o-mini (especially zero-shot) had the fewest failure modes of any model tested: virtually zero near-copies, zero bloat, and zero info-loss cases. Every output fell within the 0.75–1.15 length ratio band. Its only failure mode was occasionally making text harder (6 examples), most of them marginal. If an API model is needed for deployment simplicity or as a fallback, zero-shot GPT-4o-mini is a reasonable choice.

### Few-shot prompting

Across both GPT models, few-shot prompting didn't help:


| Model       | Zero-shot FKGL | Few-shot FKGL | Few-shot lower on |
| ----------- | -------------- | ------------- | ----------------- |
| GPT-4o-mini | 11.5           | 12.0          | 15/30             |
| GPT-5.4     | 12.8           | 12.0          | 15/30             |


The 15/30 splits are coin flips. The system prompt alone is sufficient for the API baselines.

## 4. Failure modes

### 4.1 Near-copies (change rate < 10%)


| Model                                 | Count |
| ------------------------------------- | ----- |
| FT Llama 3.1-8B                       | 10/30 |
| Few-shot GPT-5.4                      | 4/30  |
| Zero-shot GPT-5.4                     | 4/30  |
| FT Qwen 2.5-7B                        | 3/30  |
| Zero-shot GPT-4o-mini                 | 1/30  |
| Base Qwen, Base Llama, FS GPT-4o-mini | 0/30  |


A few examples already written in relatively plain language (FKGL 6.3, 7.4, and 9.7) trigger near-copies across multiple models. But the task asks for a plain-English rewrite, so keeping the text the same doesn’t fulfill it.

### 4.2 Bloat (length ratio > 1.15)

Base Llama was the worst offender at 11/30 examples with bloated output. Fine-tuning eliminated this almost entirely for both models. The API models showed near-perfect length control.

### 4.3 Universally hard examples

In a citation-heavy legal shorthand example (input FKGL 17.5), four of eight model variants produced output harder than the input. Citation formatting resists simplification because abbreviations and parenthetical references inflate FKGL when expanded.

An informal HR email example (input FKGL 8.5) was already at a low grade level. Most models made it harder by formalizing the casual tone, which increases sentence length.

## 5. Metric limitations

FKGL measures surface readability (word length, sentence length), not comprehension. Sentence-splitting being the main simplification move is partly circular, since FKGL directly rewards shorter sentences. Whether sentence-splitting actually improves comprehension for legal text consumers is an empirical question this evaluation cannot answer.

FKGL + change rate + length ratio capture surface readability, rewriting effort, and information quantity. They don't capture semantic faithfulness (did the model preserve the legal meaning?) or simplification quality (does the rewrite match what a human would produce?). BERTScore on input-output pairs for faithfulness, and output-gold for quality would address both gaps. A small human evaluation would provide ground truth on if FKGL improvements correspond to actual comprehensibility gains.

## 6. Takeaways

- **Fine-tuned Qwen 2.5-7B is the recommended model for this pipeline.** It matches the best API models on readability, produces the most consistently well-sized outputs, and runs locally. Its main weakness, 3 near-copies, is minor compared to alternatives. Its advantage over zero-shot GPT-4o-mini is small enough that the choice is as much about deployment constraints (cost, privacy, latency) as about quality.
- **Domain-adapted fine-tuning pays off at small scale.** 350 training examples with QLoRA is enough to match or exceed API models on this task.
- **Few-shot prompting wasn't worth the effort here.** It didn't reliably improve either GPT model. The system prompt alone is sufficient.

## 7. What's next?

- Add BERTScore F1 for both input-output (faithfulness) and output-gold reference (quality) pairs
- Try `completion_only_loss=True` ablation for FT models and retrain.
- Expand test set to 50–100 examples for better statistical power
- Human evaluation on a subset
- Test an open-weight 70B model & Gemini 2.5 Flash
