import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import (
    DATASET_DIR, MODEL_OUTPUT_DIR, BASE_MODEL,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LENGTH,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

if device == "cuda":
    try:
        gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gb}GB")

        if gb >= 18:
            MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"
        elif gb >= 12:
            MODEL_NAME = "unsloth/Qwen2.5-7B-bnb-4bit"
        else:
            MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"

        print(f"Modelo seleccionado: {MODEL_NAME}")
        USE_UNSLOTH = True
    except:
        MODEL_NAME = BASE_MODEL
        USE_UNSLOTH = False
else:
    MODEL_NAME = BASE_MODEL
    USE_UNSLOTH = False

if USE_UNSLOTH:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model.print_trainable_parameters()
else:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if device == "cuda":
        print(f"Cargando {MODEL_NAME} con 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Cargando {MODEL_NAME} en CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

def tokenize_fn(examples):
    texts = examples["text"]
    encodings = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_file = DATASET_DIR / "train.jsonl"
val_file = DATASET_DIR / "val.jsonl"
if not train_file.exists():
    print("Dataset no encontrado. Ejecuta prepare_dataset.py primero.")
    sys.exit(1)

dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})
print(f"Dataset: {len(dataset['train'])} train, {len(dataset['validation'])} val")

if USE_UNSLOTH:
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=LEARNING_RATE,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
    )

    trl_ver = __import__('trl').__version__
    major, minor = map(int, trl_ver.split('.')[:2])
    train_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    if major >= 1 or (major == 0 and minor >= 12):
        train_kwargs['processing_class'] = tokenizer
    else:
        train_kwargs['tokenizer'] = tokenizer

    trainer = SFTTrainer(**train_kwargs)
else:
    from transformers import Trainer, DataCollatorForSeq2Seq, TrainingArguments
    from functools import partial

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
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
        fp16=True,
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

print("Iniciando fine-tuning elite...")
trainer.train()

output_dir = str(MODEL_OUTPUT_DIR)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo guardado en {output_dir}")

if USE_UNSLOTH:
    print("\nConvirtiendo a GGUF para Ollama...")
    from unsloth import save_to_gguf
    save_to_gguf(output_dir, output_dir / "fsociety-v2-q4_k_m.gguf", outtype="q4_k_m")
    print(f"GGUF guardado en {output_dir / 'fsociety-v2-q4_k_m.gguf'}")
    print("\nComando para crear modelo Ollama:")
    print(f"  echo 'FROM {output_dir}/fsociety-v2-q4_k_m.gguf' > Modelfile")
    print("  ollama create fsociety-v2 -f Modelfile")
