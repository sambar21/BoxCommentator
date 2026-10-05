"""
LoRA fine-tune Qwen2.5-1.5B-Instruct on the distilled commentary data
(Hugging Face Transformers + PEFT). Meant for a Kaggle T4.

  python -m finetune.train_lora --data finetune/data --out /kaggle/working/boxer_lora

Design choices
  - Loss only on the assistant reply (prompt tokens are masked with -100), so the model
    learns to *write commentary*, not to reproduce the long prompt.
  - Base weights are loaded in fp32 and run under fp16 autocast: T4 has no bf16, and fp16
    base weights + fp16 LoRA params make the gradient scaler fail. 1.5B in fp32 is ~6 GB.
  - Adapter only is saved (a few MB); vLLM serves it with --enable-lora.

Heavy imports (torch, transformers, peft) are inside functions so the data-handling code
can be imported and tested without a GPU stack.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_example(tokenizer, messages: List[dict], max_len: int = 1536) -> Dict[str, List[int]]:
    """
    Tokenize one chat example with the assistant reply as the only training target.

    The prompt part is rendered with the model's own chat template (plus generation prompt) so
    training sees exactly what vLLM will send at inference; the reply and the end-of-turn
    token are appended after it.
    """
    user_msgs, reply = messages[:-1], messages[-1]["content"]
    # Render to text, then tokenize: apply_chat_template(tokenize=True) returns a list in older
    # transformers and a BatchEncoding (not a dict) in newer ones.
    prompt_text = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(reply + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + list(reply_ids)
    labels = [-100] * len(prompt_ids) + list(reply_ids)
    if len(input_ids) > max_len:
        # keep the END of the prompt (the live situation) and the full reply
        cut = len(input_ids) - max_len
        input_ids, labels = input_ids[cut:], labels[cut:]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def collate(batch: List[dict], pad_id: int):
    import torch
    width = max(len(b["input_ids"]) for b in batch)

    def pad(key, value):
        return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

    return {"input_ids": pad("input_ids", pad_id), "labels": pad("labels", -100),
            "attention_mask": pad("attention_mask", 0)}


def train(args) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    data_dir = Path(args.data)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    eval_rows = load_jsonl(data_dir / "eval.jsonl")
    train_ds = [build_example(tokenizer, r["messages"], args.max_len) for r in train_rows]
    eval_ds = [build_example(tokenizer, r["messages"], args.max_len) for r in eval_rows]
    print(f"train {len(train_ds)}  eval {len(eval_ds)}  "
          f"max tokens {max(len(x['input_ids']) for x in train_ds)}")

    # fp32 must be explicit: transformers 5 defaults to the checkpoint's bf16, which a T4 can't train in.
    # The kwarg is `dtype` in new versions and `torch_dtype` in old ones.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # warmup_ratio was removed in transformers 5; warmup_steps works in old and new versions.
    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, round(0.05 * steps_per_epoch * args.epochs))

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(), logging_steps=5,
        per_device_eval_batch_size=1, prediction_loss_only=True,  # 151k-vocab logits OOM a T4 at eval batch 8
        eval_strategy="epoch", save_strategy="no", report_to="none",
        remove_unused_columns=False, seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=lambda b: collate(b, tokenizer.pad_token_id))
    trainer.train()
    metrics = trainer.evaluate()

    out = Path(args.out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    log = {"base": args.base, "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs,
           "lr": args.lr, "train_examples": len(train_ds), "eval_examples": len(eval_ds),
           "eval_loss": metrics.get("eval_loss"),
           "log_history": trainer.state.log_history}
    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"adapter saved to {out}  (eval loss {metrics.get('eval_loss')})")
    return log


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).parent / "data"))
    p.add_argument("--out", default="boxer_lora")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--seed", type=int, default=7)
    train(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
