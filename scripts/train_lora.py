#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for MEOKCLAW Domain Model.
Optimized for 12GB VRAM (RTX 4070 SUPER) using QLoRA.
Can also run on RunPod Serverless for bigger models.
"""
import os
import sys
from pathlib import Path

# Config
MODEL_NAME = os.environ.get("FT_MODEL", "unsloth/Qwen2.5-7B-Instruct")
DATA_DIR = Path("/Users/nicholas/clawd/sovereign-temple/data/finetune")
OUTPUT_DIR = Path("/Users/nicholas/clawd/sovereign-temple/data/finetune_output")
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 10
LOGGING_STEPS = 10
SAVE_STEPS = 100


def train_with_unsloth():
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install unsloth transformers trl datasets")
        sys.exit(1)

    print(f"🚀 Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl")}, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            max_steps=-1,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            logging_steps=LOGGING_STEPS,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(OUTPUT_DIR),
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            report_to="none",
        ),
    )

    print("🏋️ Starting training...")
    trainer_stats = trainer.train()
    print(f"✅ Training complete! Loss: {trainer_stats.training_loss:.4f}")

    # Save adapter
    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"💾 Adapter saved to {adapter_dir}")

    # Save merged model (optional, needs more VRAM)
    try:
        merged_dir = OUTPUT_DIR / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"💾 Merged model saved to {merged_dir}")
    except Exception as e:
        print(f"⚠️ Merged save skipped: {e}")

    # Export to GGUF for Ollama
    try:
        gguf_dir = OUTPUT_DIR / "gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
        print(f"💾 GGUF saved to {gguf_dir}")
    except Exception as e:
        print(f"⚠️ GGUF export skipped: {e}")


def train_with_trl():
    """Fallback using trl + peft without unsloth."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
        import torch
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install transformers peft trl datasets bitsandbytes accelerate")
        sys.exit(1)

    print(f"🚀 Loading model (TRL fallback): {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl")}, split="train")

    def format_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=LOGGING_STEPS,
            output_dir=str(OUTPUT_DIR),
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            report_to="none",
        ),
    )

    print("🏋️ Starting training...")
    trainer.train()
    print("✅ Training complete!")

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"💾 Adapter saved to {adapter_dir}")


def main():
    print("=" * 60)
    print("MEOKCLAW LoRA FINE-TUNING")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATA_DIR}/train.jsonl")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}x{GRAD_ACCUM}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try unsloth first, fallback to trl
    try:
        train_with_unsloth()
    except ImportError:
        print("⚠️ unsloth not available, using trl fallback")
        train_with_trl()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Copy GGUF to Ollama models directory:")
    print(f"   cp {OUTPUT_DIR}/gguf/* /path/to/ollama/models/")
    print("2. Or load adapter with Ollama:")
    print(f"   ollama create meokclaw-lora -f Modelfile")
    print("3. Or use with vLLM/TGI for API serving")


if __name__ == "__main__":
    main()
