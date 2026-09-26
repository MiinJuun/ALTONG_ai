"""Run a small Qwen3-0.6B LoRA SFT smoke training job."""

import argparse
import hashlib
import json
from pathlib import Path


PREPARED_DIR = Path(__file__).resolve().parent / "outputs" / "prepared"
RUN_DIR = Path(__file__).resolve().parent / "outputs" / "lora-smoke"
MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_training_pairs(path: Path, tokenizer, max_length: int) -> list[dict[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            messages = record["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"invalid message roles at line {line_number}")
            prompt = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            completion = messages[2]["content"] + tokenizer.eos_token
            length = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
            if length > max_length:
                raise ValueError(
                    f"sample at line {line_number} has {length} tokens; "
                    f"increase --max-length above {max_length} to avoid truncation"
                )
            pairs.append({"prompt": prompt, "completion": completion})
    if not pairs:
        raise ValueError("training split is empty")
    return pairs


def train(prepared_dir: Path, run_dir: Path, max_steps: int,
          max_length: int, seed: int, model_name: str) -> dict:
    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local LoRA smoke run")
    if max_steps < 1 or max_length < 1:
        raise ValueError("max_steps and max_length must be positive")
    set_seed(seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pairs = load_training_pairs(prepared_dir / "train.jsonl", tokenizer, max_length)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": model_name,
        "dataset_sha256": manifest["source_sha256"],
        "prepared_manifest_sha256": hashlib.sha256(
            (prepared_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "train_count": len(pairs),
        "max_steps": max_steps,
        "max_length": max_length,
        "seed": seed,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05,
                 "target_modules": ["q_proj", "v_proj"]},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__},
        "device": torch.cuda.get_device_name(0),
        "precision": "bf16" if use_bf16 else "fp16",
        "purpose": "pipeline smoke test, not model quality evidence",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if use_bf16 else torch.float16)
    model.config.use_cache = False
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        max_steps=max_steps,
        max_length=max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        optim="adamw_torch",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
        completion_only_loss=True,
        assistant_only_loss=False,
        eos_token=tokenizer.eos_token,
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=lora,
    )
    prepared_sample = trainer.train_dataset[0]
    labels = prepared_sample.get("labels")
    if (labels is None or labels[0] != -100
            or not any(value == -100 for value in labels)
            or not any(value != -100 for value in labels)):
        raise RuntimeError("training data does not mask the prompt from loss")
    result = trainer.train()
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    run_config["training_loss"] = result.training_loss
    run_config["adapter_dir"] = str(adapter_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = train(args.prepared_dir, args.run_dir, args.max_steps,
                   args.max_length, args.seed, args.model)
    print(json.dumps({"training_loss": config["training_loss"],
                      "adapter_dir": config["adapter_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
