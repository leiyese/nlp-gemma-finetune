#!/usr/bin/env python3
"""
Training script: BERT (small) fine-tuning entry point.

This is a scaffold. Fill in dataset paths, model name, and hyperparameters
as needed in later steps.

Usage (fish/bash):
  python scripts/train_bert_small.py \
    --model-name prajjwal1/bert-small \
    --train-file data/train.csv \
    --eval-file data/val.csv \
    --test-file data/test.csv \
    --output-dir outputs/bert-small \
    --epochs 3 \
    --batch-size 32
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Step 2 imports: models/tokenizer, datasets, torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset
import torch  # noqa: F401  # used later when we build torch datasets / models
import re
import inspect
import evaluate
import time
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small BERT model on your dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="prajjwal1/bert-small",
        help="HF model checkpoint to start from (e.g., prajjwal1/bert-small)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.csv",
        help="Path to training data file (e.g., CSV/JSON)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/val.csv",
        help="Path to evaluation/validation data file",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/test.csv",
        help="Path to test data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bert-small",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Name of the text column in the CSV files",
    )
    parser.add_argument("--freeze-layers", type=int, default=2, help="How many initial encoder layers to freeze (default: 2)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler (filtered if unsupported)")
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="LR scheduler type, e.g. linear, cosine (filtered if unsupported)",
    )
    parser.add_argument("--class-weight", action="store_true", help="Enable class weighting to handle imbalance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: print config so we can verify wiring in step-by-step flow
    print("[train_bert_small] Configuration:")
    print(f"  model_name:  {args.model_name}")
    print(f"  train_file:  {args.train_file or '(not set)'}")
    print(f"  eval_file:   {args.eval_file or '(not set)'}")
    print(f"  test_file:   {args.test_file or '(not set)'}")
    print(f"  output_dir:  {out_dir}")
    print(f"  epochs:      {args.epochs}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  lr:          {args.lr}")
    print(f"  seed:        {args.seed}")

    # Step 2 — Importera bibliotek och ladda dataset
    # 1) Välj modell och ladda tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("[train_bert_small] Tokenizer loaded.")

    # 2) Ladda dina färdiga datafiler (CSV)
    data_files = {
        "train": args.train_file,
        "validation": args.eval_file,
        "test": args.test_file,
    }

    # Only proceed if files exist to avoid crashing when paths are placeholders
    missing = [k for k, p in data_files.items() if not (p and Path(p).exists())]
    if missing:
        print(
            "[train_bert_small] Warning: Missing data files for splits:", ", ".join(missing)
        )
        print(
            "Provide CSV files and rerun, or override with --train-file/--eval-file/--test-file."
        )
        print(
            f"Expected columns include '{args.text_column}'. Tokenization will run once files exist."
        )
        print("[train_bert_small] Exiting after configuration stage.")
        return 0

    dataset = load_dataset("csv", data_files=data_files)
    print(dataset)

    # 3) Tokenisera data
    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("[train_bert_small] Tokenization complete.")

    # Step 3 — Ladda modellen och frys tidiga lager
    # 5) Ladda modellen (antag 3 klasser)
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    print("[train_bert_small] Model loaded.")

    # 6) Frys tidiga lager: de första N encoder-lagren
    frozen = 0
    pattern = re.compile(r"^(?:bert\.|)(?:encoder\.layer\.(\d+)\.|embeddings\.)")
    for name, param in model.named_parameters():
        m = pattern.match(name)
        if m:
            layer_idx = m.group(1)
            # Freeze embeddings always; and freeze first N layers by index
            n_freeze = max(0, int(args.freeze_layers))
            if layer_idx is None or (layer_idx.isdigit() and int(layer_idx) < n_freeze):
                param.requires_grad = False
                frozen += 1

    print(f"[train_bert_small] Frozen parameters in first {int(args.freeze_layers)} layer(s): {frozen}")

    # Calculate class weights if enabled
    if args.class_weight:
        # Compute class weights based on train dataset labels
        labels = tokenized_datasets["train"]["label_id"]  # Assuming 'label_id' is the column with integer labels
        classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)  # Move to device if using GPU
        print(f"[train_bert_small] Class weights computed: {class_weights}")
    else:
        class_weights = None

    # Step 4 — Lägg till träningsinställningar (TrainingArguments)
    models_dir = Path("models/bert-small-finance")
    logs_dir = Path("logs")
    status_file = Path("logs/training_status.txt")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build kwargs and filter by TrainingArguments signature for version compatibility
    ta_kwargs = {
        "output_dir": str(models_dir),
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": str(logs_dir),
        "logging_steps": 10,
        "logging_strategy": "steps",
        "disable_tqdm": False,
        "report_to": ["none"],
        # Scheduler and warmup (filtered if unsupported)
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        # Gradient clipping for stability (filtered if unsupported)
        "max_grad_norm": 1.0,
        # Best model tracking (will be filtered if unsupported)
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,
        # Avoid pin_memory warning on CPU-only setups (filtered if unsupported)
        "dataloader_pin_memory": False,
    }
    ta_sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    # remove 'self' if present
    ta_sig.discard("self")
    # If this version does NOT support evaluation_strategy, disable best-model options to avoid runtime mismatch
    if "evaluation_strategy" not in ta_sig:
        for k in [
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
            "save_total_limit",
        ]:
            ta_kwargs.pop(k, None)
    filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_sig}
    training_args = TrainingArguments(**filtered_kwargs)
    print("[train_bert_small] TrainingArguments configured.")

    # Prepare datasets for Trainer: ensure 'labels' column exists and set torch format
    if "label_id" in tokenized_datasets["train"].column_names and "labels" not in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")

    # Remove unused columns to avoid passing raw text/other fields to the model forward
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split in tokenized_datasets.keys():
        cols = tokenized_datasets[split].column_names
        remove_cols = [c for c in cols if c not in keep_cols]
        if remove_cols:
            tokenized_datasets[split] = tokenized_datasets[split].remove_columns(remove_cols)
    # Set torch format
    for split in tokenized_datasets.keys():
        present = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in tokenized_datasets[split].column_names]
        tokenized_datasets[split].set_format(type="torch", columns=present)

    # Metrics: accuracy + macro F1
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        import numpy as np
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # Progress indicator callback
    class StatusCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            status_file.parent.mkdir(parents=True, exist_ok=True)
            with status_file.open("w") as f:
                f.write(f"TRAINING_STARTED {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print("[train_bert_small] Training started…")

        def on_step_end(self, args, state, control, **kwargs):
            # state.global_step, state.epoch, state.log_history may contain loss
            msg = f"STEP {state.global_step} EPOCH {getattr(state, 'epoch', 0):.2f}"
            with status_file.open("a") as f:
                f.write(msg + "\n")
            # Light console indicator every 'logging_steps'
            if state.global_step % max(1, getattr(args, 'logging_steps', 10)) == 0:
                print(f"[train_bert_small] {msg}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with status_file.open("a") as f:
                    f.write("LOG " + ", ".join(f"{k}={v}" for k, v in logs.items()) + "\n")

        def on_train_end(self, args, state, control, **kwargs):
            with status_file.open("a") as f:
                f.write(f"TRAINING_FINISHED {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print("[train_bert_small] Training finished.")

    if args.epochs > 0:
        if args.class_weight:
            # Define and use WeightedTrainer for training with class weights
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                    loss = loss_fct(logits, labels)
                    return (loss, outputs) if return_outputs else loss
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=compute_metrics,
                callbacks=[StatusCallback()],  # Include any callbacks
            )
            print(f"[train_bert_small] Using WeightedTrainer with class weights for training: {class_weights}")
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=compute_metrics,
                callbacks=[StatusCallback()],  # Include any callbacks
            )
        # Run training
        trainer.train()
    else:
        # For evaluation only (e.g., --epochs 0), use standard Trainer with no train_dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            callbacks=[StatusCallback()],  # Include any callbacks for consistency
        )
        print("[train_bert_small] Running evaluation only...")
        trainer.evaluate(eval_dataset=tokenized_datasets["validation"])

    # Always run an explicit evaluation so you see metrics even if evaluation_strategy wasn't supported
    try:
        print("[train_bert_small] Running final evaluation...")
        final_metrics = trainer.evaluate()
        print("[train_bert_small] Eval metrics:", final_metrics)
    except Exception as e:
        print(f"[train_bert_small] Final evaluation skipped due to error: {e}")

    # Additionally evaluate on the test split for an unbiased estimate
    try:
        print("[train_bert_small] Running test evaluation...")
        test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print("[train_bert_small] Test metrics:", test_metrics)
    except Exception as e:
        # Fallback for very old versions that may not support named argument or metric_key_prefix
        try:
            test_metrics = trainer.evaluate(tokenized_datasets["test"])  # type: ignore[arg-type]
            print("[train_bert_small] Test metrics:", test_metrics)
        except Exception as e2:
            print(f"[train_bert_small] Test evaluation skipped due to error: {e2}")

    # TODO: In next steps we will:
    #  1) Load and preprocess dataset
    #  2) Tokenize with BERT tokenizer
    #  3) Build DataLoader(s)
    #  4) Initialize model, optimizer, scheduler
    #  5) Train and evaluate per epoch
    #  6) Save best checkpoint/metrics to output_dir

    print("[train_bert_small] Scaffold ready. Implement training in the next step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
