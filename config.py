from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIRS = [
    BASE_DIR / "CIBERSEGURIDAD.TODOS-LOS-PDF",
]
PDF_FILES = [
    BASE_DIR / "00186_R3v3rsing_Ingenieria_nversa.pdf",
    BASE_DIR / "iOSAppReverseEngineering.pdf",
    BASE_DIR / "Kurose-7.pdf",
    BASE_DIR / "Libro-Hacking-Etico-Teoria-Practicas.pdf",
    BASE_DIR / "libro_seguridad.pdf",
    BASE_DIR / "memoria.pdf",
    BASE_DIR / "modernC.pdf",
    BASE_DIR / "RE4B-ES (1).pdf",
    BASE_DIR / "RE4B-ES.pdf",
]

EXTRACTS_DIR = BASE_DIR / "Entrenamiento" / "extracts"
CHUNKS_DIR = BASE_DIR / "Entrenamiento" / "chunks"
VECTOR_DB_DIR = BASE_DIR / "Entrenamiento" / "vector_db"
DATASET_DIR = BASE_DIR / "Entrenamiento" / "dataset"
MODEL_NAME = "fsociety-v2"
MODEL_OUTPUT_DIR = Path("D:/LLMsFinetunnig") / MODEL_NAME

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TURBOVEC_BIT_WIDTH = 2
TURBOVEC_INDEX_PATH = VECTOR_DB_DIR / "turbovec.tvim"
TURBOVEC_METADATA_PATH = VECTOR_DB_DIR / "metadata.json"

BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048

# TurboVec config
TURBOVEC_BIT_WIDTH = 2

# Anti-refusal: permite respuestas sin restricciones en seguridad ofensiva
ANTI_REFUSAL = True
