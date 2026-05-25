import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import (
    DATASET_DIR, MODEL_OUTPUT_DIR, BASE_MODEL,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LENGTH,
)

def tokenize_function(examples, tokenizer):
    texts = examples["text"]
    model_inputs = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def main():
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"
    if not train_file.exists():
        print("Dataset no encontrado. Ejecuta prepare_dataset.py primero.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    if device == "cuda":
        print(f"Cargando {BASE_MODEL} con 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Cargando {BASE_MODEL} en CPU (GPU no disponible)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    print("Aplicando LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Cargando dataset...")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=LEARNING_RATE,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Iniciando fine-tuning...")
    trainer.train()

    print(f"Guardando modelo en {MODEL_OUTPUT_DIR}")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    print("Fine-tuning completado!")

    print(f"\nPara convertir a GGUF usa:")
    print(f"  python ConvertEverywhere.py {MODEL_OUTPUT_DIR} --outfile fsociety-v2-q4_k_m.gguf --outtype q4_k_m")

if __name__ == "__main__":
    main()
