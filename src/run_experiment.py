import os
import re
import gc
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def prepare_esg_dataset(seed: int = 42):
    """Loads and merges ESGBERT E/S/G datasets into a balanced multi-class split."""
    subsets = [
        ("ESGBERT/environmental_2k", "Environmental"),
        ("ESGBERT/social_2k", "Social"),
        ("ESGBERT/governance_2k", "Governance"),
    ]
    splits_train, splits_test = [], []
    for ds_name, label_str in subsets:
        ds = load_dataset(ds_name)
        train_part = ds["train"].filter(lambda x: x["label"] == 1)
        test_part = ds["test"].filter(lambda x: x["label"] == 1)
        train_part = train_part.map(lambda x: {"text": x["text"], "target": label_str}, remove_columns=train_part.column_names)
        test_part = test_part.map(lambda x: {"text": x["text"], "target": label_str}, remove_columns=test_part.column_names)
        splits_train.append(train_part)
        splits_test.append(test_part)

    full_train = concatenate_datasets(splits_train).shuffle(seed=seed)
    val_split = full_train.train_test_split(test_size=0.15, seed=seed)
    test_ds = concatenate_datasets(splits_test).shuffle(seed=seed)
    return DatasetDict({
        "train": val_split["train"],
        "validation": val_split["test"],
        "test": test_ds,
    })


def prepare_folio_dataset(seed: int = 42):
    """Loads yale-nlp/FOLIO and formats premises + conclusion -> True/False/Unknown."""
    ds = load_dataset("yale-nlp/FOLIO")
    def format_row(ex):
        premises = ex["premises"] if isinstance(ex["premises"], str) else " ".join(ex["premises"])
        prompt_text = f"Premises:\n{premises}\n\nConclusion:\n{ex['conclusion']}"
        return {"text": prompt_text, "target": ex["label"].strip()}

    train_full = ds["train"].map(format_row, remove_columns=ds["train"].column_names).shuffle(seed=seed)
    split_tv = train_full.train_test_split(test_size=0.15, seed=seed)
    test_ds = ds["validation"].map(format_row, remove_columns=ds["validation"].column_names)
    return DatasetDict({
        "train": split_tv["train"],
        "validation": split_tv["test"],
        "test": test_ds,
    })


def build_prompt(task: str, text: str) -> str:
    if task == "esg":
        return (
            "You are a financial and corporate sustainability analyst. "
            "Classify the following corporate statement into exactly one category: "
            "Environmental, Social, or Governance.\n\n"
            f"Statement: {text}\n"
            "Category:"
        )
    elif task == "folio":
        return (
            "You are a formal logic reasoner. Based strictly on the given premises, "
            "determine whether the conclusion is True, False, or Unknown. "
            "Respond with exactly one word: True, False, or Unknown.\n\n"
            f"{text}\n"
            "Answer:"
        )
    raise ValueError(f"Unsupported task: {task}")


def normalize_prediction(raw_text: str, valid_labels: list) -> str:
    cleaned = raw_text.strip().split("\n")[0].strip()
    cleaned = re.sub(r"[^A-Za-z]", "", cleaned).lower()
    for lbl in valid_labels:
        if cleaned.startswith(lbl.lower()):
            return lbl
    for lbl in valid_labels:
        if lbl.lower() in raw_text.lower():
            return lbl
    return "INVALID"


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(mask):
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * np.mean(mask)
    return float(ece)


def load_model_and_tokenizer(model_id: str, method: str, cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method_cfg = cfg["methods"][method]
    quant_config = None
    if method_cfg.get("load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if method_cfg["type"] == "peft":
        if method_cfg.get("load_in_4bit", False):
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=method_cfg["r"],
            lora_alpha=method_cfg["lora_alpha"],
            lora_dropout=method_cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=method_cfg.get("use_dora", False),
        )
        model = get_peft_model(model, peft_config)
    elif method_cfg["type"] == "full":
        if cfg["training_defaults"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()

    return model, tokenizer


def run(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_id = cfg["models"][args.model_key]
    task_cfg = cfg["tasks"][args.task]
    method_cfg = cfg["methods"][args.method]
    valid_labels = task_cfg["labels"]
    max_len = task_cfg["max_seq_length"]

    ds = prepare_esg_dataset(args.seed) if args.task == "esg" else prepare_folio_dataset(args.seed)
    if args.max_samples > 0:
        ds["train"] = ds["train"].select(range(min(args.max_samples, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(max(50, args.max_samples // 5), len(ds["validation"]))))
        ds["test"] = ds["test"].select(range(min(max(100, args.max_samples // 3), len(ds["test"]))))

    model, tokenizer = load_model_and_tokenizer(model_id, args.method, cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_wall_time_sec = 0.0
    if method_cfg["type"] != "baseline":
        def tokenize_sft(example):
            prompt = build_prompt(args.task, example["text"])
            target = " " + example["target"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len - 16, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(target, truncation=True, max_length=16, add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

        tokenized_train = ds["train"].map(tokenize_sft, remove_columns=ds["train"].column_names)
        tokenized_val = ds["validation"].map(tokenize_sft, remove_columns=ds["validation"].column_names)

        out_dir = os.path.join(args.output_dir, f"checkpoints_{args.task}_{args.method}_seed{args.seed}")
        t_cfg = cfg["training_defaults"]
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=task_cfg["num_epochs"],
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=t_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            learning_rate=method_cfg["lr"],
            warmup_ratio=t_cfg["warmup_ratio"],
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            weight_decay=t_cfg["weight_decay"],
            bf16=t_cfg["bf16"],
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        t0 = time.perf_counter()
        trainer.train()
        train_wall_time_sec = time.perf_counter() - t0

    # Evaluation on Test Set
    model.eval()
    preds, golds, confidences = [], [], []
    label_token_ids = {lbl: tokenizer.encode(" " + lbl, add_special_tokens=False)[0] for lbl in valid_labels}

    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for ex in ds["test"]:
            prompt = build_prompt(args.task, ex["text"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_label = normalize_prediction(gen_text, valid_labels)

            first_step_logits = outputs.scores[0][0]
            probs = torch.softmax(first_step_logits.float(), dim=-1)
            cand_probs = [probs[tid].item() for tid in label_token_ids.values()]
            norm_conf = max(cand_probs) / (sum(cand_probs) + 1e-12)

            preds.append(pred_label)
            golds.append(ex["target"])
            confidences.append(norm_conf)

    infer_time_sec = time.perf_counter() - t_inf_start
    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro", labels=valid_labels, zero_division=0)
    weighted_f1 = f1_score(golds, preds, average="weighted", labels=valid_labels, zero_division=0)
    invalid_rate = sum(1 for p in preds if p == "INVALID") / len(preds)
    correctness = np.array([1.0 if p == g else 0.0 for p, g in zip(preds, golds)])
    ece = compute_ece(np.array(confidences), correctness)

    summary = {
        "model": model_id,
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(100.0 * trainable_params / total_params, 4),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "train_wall_time_sec": round(train_wall_time_sec, 2),
        "infer_time_sec": round(infer_time_sec, 2),
        "samples_per_sec": round(len(ds["test"]) / max(infer_time_sec, 1e-6), 2),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "ece": round(float(ece), 4),
        "invalid_output_rate": round(float(invalid_rate), 4),
        "classification_report": classification_report(golds, preds, labels=valid_labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(golds, preds, labels=valid_labels).tolist(),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    res_file = os.path.join(args.output_dir, f"res_{args.model_key}_{args.task}_{args.method}_seed{args.seed}.json")
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k not in ["classification_report", "confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.json")
    parser.add_argument("--model_key", type=str, default="primary", choices=["primary", "secondary"])
    parser.add_argument("--task", type=str, required=True, choices=["esg", "folio"])
    parser.add_argument("--method", type=str, required=True, choices=["zero_shot", "qlora", "lora", "dora", "full_ft"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1, help="Use >0 for quick sanity check")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run(args)
