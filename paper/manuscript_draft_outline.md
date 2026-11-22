# Parameter-Efficient Fine-Tuning Across Cognitive Regimes: A Controlled Empirical Comparison of QLoRA, LoRA, DoRA, and Full Fine-Tuning on ESG Classification and Deductive Logical Reasoning

## Abstract
We present a controlled empirical study comparing parameter-efficient fine-tuning (PEFT) methods—4-bit NF4 QLoRA, LoRA, and DoRA—against zero-shot prompting and full fine-tuning across two distinct cognitive regimes: domain-specific sustainability/ESG classification (ESGBERT) and multi-step deductive logical reasoning (FOLIO). Across multiple random seeds and hardware profiles on Google Colab Pro (A100/L4), we jointly evaluate task performance, peak VRAM allocation, wall-clock training duration, inference throughput, calibration error (ECE), and out-of-domain retention.

## 1. Introduction
- Context: Rapid proliferation of 7B–9B instruction-tuned LLMs and the practical cost of adapting them under constrained GPU memory budgets.
- Gap: Prior PEFT benchmarks largely evaluate single-task regimes or omit joint analysis of accuracy, VRAM/latency Pareto trade-offs, and probability calibration.
- Research Questions: RQ1 (Cognitive regime performance), RQ2 (Compute/VRAM/latency Pareto frontier), RQ3 (Catastrophic forgetting & calibration).

## 2. Related Work
- Low-rank and weight-decomposed adaptation (LoRA, QLoRA, DoRA).
- Domain adaptation in financial/ESG NLP vs. formal/deductive reasoning benchmarks.

## 3. Methodology & Experimental Protocol
- Backbones: Qwen/Qwen2.5-7B-Instruct (primary) and meta-llama/Llama-3.1-8B-Instruct (secondary check).
- Datasets & Splits: ESGBERT (Environmental, Social, Governance) and FOLIO (True, False, Unknown).
- Controlled Hyperparameters & Seeds: Fixed splits and seeds (42, 1024, 2025).

## 4. Results & Discussion (To Be Populated from Colab Runs)
- Table 1: Task performance (Accuracy, Macro-F1, ECE) across methods and seeds.
- Table 2: Compute & memory profile (Trainable %, Peak VRAM GB, Training Time, Throughput).
- Figure 1: Accuracy vs. Peak VRAM Pareto frontier.
- Figure 2: Rank ablation (r in {8, 16, 32}) and out-of-domain / forgetting analysis.

## 5. Limitations & Conclusion
- Hardware and quantization trade-offs, prompt sensitivity, and practical guidance for practitioners.
