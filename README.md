# AI-Assisted Requirements Engineering: ISO 29148 Requirement Generation

This repository contains the data processing pipeline and fine-tuning experiments for transforming vague, informal customer requirements into **ISO 29148-compliant, INCOSE-conformant requirements** using the Qwen2.5 model family (0.5B–14B). Two preference-alignment strategies are implemented and compared: **DPO** (Direct Preference Optimization) and **GRPO** (Group Relative Policy Optimization).

## Overview

| Stage | Purpose | Key files |
|---|---|---|
| 1. Data processing | Turn raw/vague customer requirements into ISO 29148-compliant target requirements using the 42 INCOSE transformation rules | `requirements_processing_with_INCOSE_rules`, `vague.xlsx` |
| 2. DPO training | Fine-tune Qwen2.5 (0.5B–14B) with preference pairs (chosen = ISO-compliant, rejected = vague) | `DPO.ipynb` |
| 3. GRPO training | Fine-tune Qwen2.5 (3B–14B) with a rule-based reward signal instead of preference pairs | `GRPO_reward_function.py`, `GRPO_train_unsloth.ipynb`, `GRPO_training_loop_rule_based.ipynb` |
| 4. Evaluation | Compare DPO vs. GRPO vs. baselines across eleven quantitative metrics | Evaluation cells in `DPO.ipynb` |

---

## 1. Data Processing

### 1.1 Source data

- **`vague.xlsx`** — Pairs of `Customer_Req` (informal, vague customer language) and `Detailed_Requirement` (ISO 29148-compliant target), e.g.:

  | Customer_Req | Detailed_Requirement |
  |---|---|
  | "The lock should be super secure and impossible to break." | "The lock system shall provide a security level of ≥AES-256 equivalent encryption strength with ≥99.99% protection against unauthorized access attempts. …" |
  | "I want to be able to unlock it with my cell phone." | "When the lock system receives a valid authentication signal from the mobile device, the lock system shall unlock the locking mechanism within 2.0 ± 0.5 seconds." |

- A larger training set (`random_data_1000_req_vague_and_neutral.jsonl`) of ~1000 vague/neutral requirement pairs, generated in a LLaMA chat-template format and parsed into `(prompt, chosen, rejected)` triples for DPO.

### 1.2 `requirements_processing_with_INCOSE_rules`

A batch pipeline (using the Anthropic API) that automatically transforms customer requirements into structured, ISO-compliant requirements:

1. **Analyze** — decide whether a requirement must be **split** into multiple atomic requirements (INCOSE Rule R18: *Single Thought Sentence*), and identify sub-capabilities and placeholders (`[PLACEHOLDER]` tokens that must be preserved verbatim).
2. **Improve / Split** — rewrite the requirement (or each split sub-requirement) to comply with the full **42 INCOSE rules** (R1–R42, covering structured statements, active voice, defined terms, vague-term removal, escape-clause removal, single-thought sentences, measurable performance with tolerances, consistent terminology, etc.), classifying each result by:
   - **Type**: Functional / Performance / Interface / Safety / Security
   - **Verification method**: Test / Inspection / Analysis / Demonstration
   - Vague terms removed, tolerances added, applicable rules
3. **Output** — a structured Excel workbook (`Category`, `Customer_Req`, `Ambiguities_Identified`, `Improvements_Made`, `Vague_Terms_Removed`, `Tolerances_Added`, `Consolidated_Requirement`, `Detailed_Requirement`, `Sub_Requirement_Text`, `Verification_Method`).

The 42 rules are grouped into 7 quality characteristics used throughout the project for both prompting and reward scoring: **Singular, Unambiguous, Complete, Feasible, Verifiable, Appropriate, Consistent.**

---

## 2. DPO Training (`DPO.ipynb`)

### 2.1 Data preparation
- Parses the JSONL chat-template dataset into `{prompt, chosen, rejected}` triples, where `chosen` is the ISO 29148-compliant requirement and `rejected` is the original vague requirement.
- 90/10 train/test split (seed=42).

### 2.2 Training setup
- **Models**: Qwen2.5-Instruct at **0.5B, 1.5B, 3B, 7B, 14B**.
- **Method**: QLoRA (4-bit NF4 quantization) + `DPOTrainer` (TRL).
- **LoRA config**: r=16 (later grid-optimized), α=32, target modules `q_proj/k_proj/v_proj/o_proj`, dropout 0.05.
- **DPO config**: 3 epochs, β=0.1 (baseline) → optimized via grid search, learning rate 5e-5 (baseline) → optimized, cosine LR schedule, warmup 100 steps, bf16.
- A full hyperparameter grid search (β, learning rate, LoRA rank, effective batch size, warmup) was run across 53 configurations to identify the optimal setting used for the final per-size models.

### 2.3 Evaluation
Each fine-tuned model is evaluated on held-out vague requirements against **eleven metrics**:

| Metric | What it measures |
|---|---|
| Mean Length | Output length in words |
| TTR (Type-Token Ratio) | Lexical diversity |
| BERTScore (P/R/F1) | Semantic similarity to reference (DeBERTa-xlarge-MNLI) |
| BLEU | N-gram overlap with reference |
| ROUGE-L | Longest common subsequence overlap |
| Perplexity | Model confidence on generated text |
| Self-BLEU | Diversity across generated outputs |
| ISO 29148 Compliance | Rule-based check (`shall`, quantifiers, numbers, system identifiers, length, units) |
| Euclidean Distance (ED) | Embedding-space distance to reference (MiniLM) |
| Semantic Matching Distance (SMD) | 1 − cosine similarity to reference |
| Cosine Similarity | Embedding similarity to reference |

Results are also compared against **un-tuned baselines** (a 405B-scale model prompted directly on vague vs. neutral requirements).

---

## 3. GRPO Training

Two parallel GRPO implementations were developed:

### 3.1 Reward functions (`GRPO_reward_function.py`)
A rule-based reward suite (no reference/preference data needed) used by both GRPO pipelines:

- `reward_xml_format` — rewards well-formed `<answer>…</answer>` output.
- `reward_iso_compliance` — full **42-rule INCOSE scoring function** (shall-language, structure/clarity, specificity/precision, verifiability/measurability, completeness, consistency, traceability, quality attributes), normalized to [0, 1].
- `reward_no_vague_terms` — penalizes vague/subjective adjectives (`appropriate`, `fast`, `robust`, etc.) with an exponential penalty.
- `reward_measurability` — rewards presence of numbers, units, ranges, and conditions.
- `reward_shall_language` — rewards `"shall"`, penalizes weak modal verbs (`should`, `may`, `will`, `might`).
- `reward_appropriate_length` — penalizes requirements that are too short (incomplete) or too long (likely compound).
- `compute_combined_reward` — weighted sum (default weights: ISO compliance 0.40, no-vague 0.15, measurability 0.15, XML format 0.10, shall-language 0.10, length 0.10).

### 3.2 TRL/Unsloth pipeline (`GRPO_train_unsloth.ipynb`)
- **Model**: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (4-bit, Unsloth-accelerated).
- **LoRA**: r=32, α=64, dropout 0.05, all projection + MLP target modules.
- **GRPO config** (`trl.GRPOConfig`): learning rate 3e-6, KL coefficient (β) 0.001, 4 generations per prompt, max prompt/completion length 512, batch size 1 with gradient accumulation 2, 500 max steps, 50 warmup steps.
- Uses the rule-based reward functions above via a robust reward wrapper, with a system prompt instructing the model to answer inside `<answer>` tags.

### 3.3 Custom from-scratch GRPO loop (`GRPO_training_loop_rule_based.ipynb`)
A from-scratch implementation of GRPO (modeled on the DeepSeek-R1 recipe) rather than relying on TRL, giving full control over group sampling, advantage computation, and KL penalty:

- **Model**: Qwen2.5-7B-Instruct.
- **Hyperparameters**: learning rate 3e-6, KL coefficient 0.001, clip ratio 10.0, temperature 1.0.
- **Group sampling**: group size 16, 32 questions/step, batch size 512.
- **Training loop**: `compute_log_probs`, `sample_group`, `compute_rewards`, `compute_advantages` (group-relative advantage — GRPO's key departure from PPO, avoiding a learned value/critic model), `compute_kl_divergence` against a periodically-updated reference model, `compute_grpo_loss`.
- **Schedule**: up to 2000 steps, reference-model sync every 400 steps, checkpointing every 400 steps, evaluation every 100 steps.

**Training history / findings:**
- A conservative configuration (v01) achieved only ~14% ISO-rule pass rate — too weak a reward signal to shift behavior meaningfully.
- A more aggressive configuration (v03) reached ~91.6% rule compliance by step 30, but then **collapsed via reward hacking** by step 70–80 (the model learned to game the rule-based reward rather than genuinely improve requirement quality).
- The **step-30 checkpoint of v03** was identified as the best-performing model overall.
- A later run with Qwen2.5-7B (8×4 = 32 generations/step) showed validation reward improving from 1.47 → 1.58 over the first 3 steps.
- ⚠️ **Known issue**: adapter checkpoints were saving at ~43 GB instead of the expected ~100–200 MB for LoRA-only weights — `save_checkpoint()` (`policy_model.save_pretrained`) is likely serializing the full base model rather than adapter-only weights. This should be fixed (e.g. ensure the model is wrapped as a PEFT model at save time, or explicitly save `get_peft_model_state_dict()`).

---

## 4. Results Summary

Evaluated on held-out vague requirements, across five DPO model sizes (0.5B–14B) and three GRPO model sizes (3B, 7B, 14B), against neutral/vague 405B-scale baselines:

- Both DPO and GRPO substantially reduce **Semantic Matching Distance** and **Euclidean Distance** to the ISO-compliant reference compared to the vague/unprompted baseline (SMD ≈ 0.30–0.38 for fine-tuned models vs. 0.39 for the vague baseline; 0.14 for the neutral baseline).
- **ISO 29148 rule-based compliance** reaches ~87–92% for DPO models and ~84–89% for GRPO models, versus much lower implicit compliance in the un-tuned baselines.
- Larger models generally show modest gains in BERTScore F1 and ROUGE-L, with diminishing returns above 7B.
- GRPO models show slightly higher generation diversity (Self-BLEU) but somewhat lower BLEU/ROUGE-L overlap with references than the best DPO models, consistent with GRPO optimizing a rule-based reward rather than direct imitation of reference text.

---

## Repository Structure

```
.
├── requirements_processing_with_INCOSE_rules   # INCOSE-based data generation pipeline (Anthropic API)
├── vague.xlsx                                  # Sample vague → ISO-compliant requirement pairs
├── DPO.ipynb                                   # Data prep, QLoRA+DPO training, evaluation (5 model sizes)
├── GRPO_reward_function.py                     # Rule-based reward suite (42 INCOSE rules)
├── GRPO_train_unsloth.ipynb                    # TRL/Unsloth-based GRPO training (Qwen2.5-3B)
└── GRPO_training_loop_rule_based.ipynb         # From-scratch GRPO implementation (Qwen2.5-7B)
```

## Requirements

```
torch, transformers, trl, datasets, accelerate, peft, bitsandbytes, unsloth
evaluate, rouge-score, nltk, sacrebleu, bert-score, sentence-transformers
pandas, openpyxl, anthropic
```

## Notes / Open Items

- Fix the LoRA-only checkpoint saving bug in the custom GRPO loop (43 GB → expected ~100–200 MB).
- Investigate GRPO reward hacking observed beyond step ~70 in the aggressive (v03) configuration; consider reward clipping, a lower learning rate after the step-30 optimum, or early stopping informed by the reward curve.
- The 42-rule reward function currently gives automatic "benefit of the doubt" points for rules that are hard to check heuristically (e.g. traceability, consistency) — these could be tightened with additional structural checks.
