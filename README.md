# Parameter-Efficient Fine-Tuning Across Cognitive Regimes
**Controlled Empirical Comparison of Zero-Shot, QLoRA, LoRA, DoRA, and Full Fine-Tuning on ESG Classification & FOLIO Deductive Reasoning**

## Repository Structure
```text
peft_study_starter/
├── configs/
│   └── experiment_config.json      # Controlled hyperparameters, seeds, and dataset definitions
├── src/
│   └── run_experiment.py           # Unified training, VRAM profiling, latency & evaluation runner
├── notebooks/
│   └── Colab_Pro_A100_Runner.ipynb # Ready-to-run Google Colab Pro notebook (Drive sync enabled)
├── results/                        # Auto-saved JSON metrics, confusion matrices & predictions
└── requirements.txt                # Reproducible Python dependencies
```

## Quick Start on Google Colab Pro (A100 40GB / L4 24GB)
1. Upload this folder to your Google Drive inside `PEFT_LLM_Comparison_Study/` (or clone from your GitHub repo).
2. Open `notebooks/Colab_Pro_A100_Runner.ipynb` in Google Colab and select **A100 GPU** (or L4 for QLoRA/LoRA/DoRA).
3. Run Phase 1 (`seed=42`) across methods (`zero_shot`, `qlora`, `lora`, `dora`, `full_ft`) for `esg` and `folio`.
4. All metrics (Macro-F1, Accuracy, Peak VRAM GB, Wall-Clock Time, Trainable Params, ECE) are saved automatically to Google Drive.
