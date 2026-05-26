import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_DIR
from datasets import load_dataset

train_file = DATASET_DIR / "train.jsonl"
val_file = DATASET_DIR / "val.jsonl"

print("Cargando dataset local...")
dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})

print(f"Subiendo a HuggingFace: {len(dataset['train'])} train + {len(dataset['validation'])} val")
dataset.push_to_hub("murdok1982/formacion-seguridad-qa-v2", private=True)
print("Dataset subido correctamente!")
