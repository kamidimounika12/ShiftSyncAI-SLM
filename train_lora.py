
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "training_config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}"
    )

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

MODEL_NAME = CONFIG["model_name"]
OUTPUT_DIR = Path(CONFIG["output_dir"])

MAX_LENGTH = int(CONFIG["max_length"])
MAX_ANSWER_LENGTH = int(CONFIG["max_answer_length"])
MAX_PROMPT_LENGTH = MAX_LENGTH - MAX_ANSWER_LENGTH

TRAIN_FILE = CONFIG["training_file"]
VALIDATION_FILE = CONFIG["validation_file"]
TEST_FILE = CONFIG.get("test_file", "test.jsonl")

REQUIRED_FIELDS = {
    "machine_id",
    "reported_issues",
    "machine_status",
    "completed_work",
    "unresolved_issues",
    "next_shift_actions",
    "missing_information",
}

LIST_FIELDS = REQUIRED_FIELDS - {
    "machine_id",
    "machine_status",
}

if MAX_LENGTH <= 0:
    raise ValueError("max_length must be positive.")

if MAX_ANSWER_LENGTH <= 0 or MAX_ANSWER_LENGTH >= MAX_LENGTH:
    raise ValueError(
        "max_answer_length must be positive and smaller than max_length."
    )

if MAX_PROMPT_LENGTH <= 0:
    raise ValueError("Prompt token budget must be positive.")


# ============================================================
# LOAD AND VALIDATE JSONL
# ============================================================

def read_jsonl(filename):
    path = BASE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {filename}, "
                    f"line {line_number}"
                ) from exc

            for key in ("instruction", "input", "output"):
                if key not in record:
                    raise ValueError(
                        f"Missing '{key}' in {filename}, "
                        f"line {line_number}"
                    )

            if not isinstance(record["instruction"], str):
                raise ValueError(
                    f"Instruction must be text in {filename}, "
                    f"line {line_number}"
                )

            if not isinstance(record["input"], str) or not record["input"].strip():
                raise ValueError(
                    f"Input must be non-empty text in {filename}, "
                    f"line {line_number}"
                )

            if not isinstance(record["output"], str):
                raise ValueError(
                    f"Output must be a JSON string in {filename}, "
                    f"line {line_number}"
                )

            try:
                report = json.loads(record["output"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON output in {filename}, "
                    f"line {line_number}"
                ) from exc

            if not isinstance(report, dict):
                raise ValueError(
                    f"Output must be one JSON object in {filename}, "
                    f"line {line_number}"
                )

            if set(report.keys()) != REQUIRED_FIELDS:
                missing = REQUIRED_FIELDS - set(report.keys())
                extra = set(report.keys()) - REQUIRED_FIELDS

                raise ValueError(
                    f"Incorrect report fields in {filename}, "
                    f"line {line_number}. "
                    f"Missing={sorted(missing)}, Extra={sorted(extra)}"
                )

            if not isinstance(report["machine_id"], str):
                raise ValueError(
                    f"machine_id must be a string in {filename}, "
                    f"line {line_number}"
                )

            if not isinstance(report["machine_status"], str):
                raise ValueError(
                    f"machine_status must be a string in {filename}, "
                    f"line {line_number}"
                )

            for field in LIST_FIELDS:
                if not isinstance(report[field], list):
                    raise ValueError(
                        f"{field} must be a list in {filename}, "
                        f"line {line_number}"
                    )

                if not all(
                    isinstance(item, str) for item in report[field]
                ):
                    raise ValueError(
                        f"{field} must contain only strings in "
                        f"{filename}, line {line_number}"
                    )

            records.append(record)

    if not records:
        raise ValueError(
            f"No training examples found in {filename}"
        )

    return records


# ============================================================
# PREPARE MODEL INPUTS
# ============================================================

def prepare_dataset(records, tokenizer):
    rows = []

    for index, record in enumerate(records, start=1):
        instruction = record["instruction"].strip()
        note = record["input"].strip()
        answer = record["output"].strip()

        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Factory shift note:\n"
            f"{note}\n\n"
            "### Structured handover report:\n"
        )

        # Keep the entire answer. Never silently truncate JSON.
        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

        if len(answer_ids) > MAX_ANSWER_LENGTH:
            raise ValueError(
                f"Example {index} answer requires "
                f"{len(answer_ids)} tokens, but the configured "
                f"limit is {MAX_ANSWER_LENGTH}. "
                "Increase max_answer_length in training_config.json."
            )

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        # Keep the end of the prompt if it exceeds its budget.
        if len(prompt_ids) > MAX_PROMPT_LENGTH:
            prompt_ids = prompt_ids[-MAX_PROMPT_LENGTH:]

        input_ids = prompt_ids + answer_ids

        # Train only on the answer, not on the prompt.
        labels = [-100] * len(prompt_ids) + answer_ids.copy()
        attention_mask = [1] * len(input_ids)

        padding = MAX_LENGTH - len(input_ids)

        if padding < 0:
            raise ValueError(
                f"Example {index} exceeds MAX_LENGTH."
            )

        input_ids += [tokenizer.pad_token_id] * padding
        labels += [-100] * padding
        attention_mask += [0] * padding

        rows.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return Dataset.from_list(rows)


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Enable the GPU accelerator "
            "in your Kaggle notebook settings."
        )

    print("=" * 60)
    print("SHIFTSYNC AI - LoRA FINE-TUNING")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base model:", MODEL_NAME)
    print("Output directory:", OUTPUT_DIR)
    print("Maximum sequence length:", MAX_LENGTH)
    print("Maximum answer length:", MAX_ANSWER_LENGTH)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading base model in FP16...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()

    # Required for gradient checkpointing with LoRA training.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("\nApplying LoRA...")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(CONFIG["lora_rank"]),
        lora_alpha=int(CONFIG["lora_alpha"]),
        lora_dropout=float(CONFIG["lora_dropout"]),
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading datasets...")

    train_records = read_jsonl(TRAIN_FILE)
    validation_records = read_jsonl(VALIDATION_FILE)

    # Load and validate the test set, but never train on it.
    test_records = read_jsonl(TEST_FILE)

    print("Training records:", len(train_records))
    print("Validation records:", len(validation_records))
    print("Test records reserved for evaluation:", len(test_records))

    tokenizer.padding_side = "right"

    train_dataset = prepare_dataset(
        train_records,
        tokenizer,
    )

    validation_dataset = prepare_dataset(
        validation_records,
        tokenizer,
    )

    print("\nPrepared datasets:")
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=int(CONFIG["epochs"]),
        per_device_train_batch_size=int(CONFIG["batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(
            CONFIG["gradient_accumulation_steps"]
        ),
        learning_rate=float(CONFIG["learning_rate"]),
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        dataloader_num_workers=0,
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=0.3,
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
    )

    print("\nStarting ShiftSync AI LoRA training...")
    training_result = trainer.train()

    print("\nTraining metrics:")
    print(training_result.metrics)

    print("\nSaving LoRA adapter...")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save training metrics for reference.
    metrics_path = OUTPUT_DIR / "training_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            training_result.metrics,
            file,
            indent=2,
            default=str,
        )

    print("\nTraining finished successfully.")
    print("Adapter directory:", OUTPUT_DIR)
    print("Metrics file:", metrics_path)
    print(
        "Next: evaluate the saved adapter on the held-out test "
        "set and separate manual safety scenarios."
    )


if __name__ == "__main__":
    main()