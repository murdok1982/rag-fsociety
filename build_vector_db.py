import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from turbovec import IdMapIndex
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import EXTRACTS_DIR, CHUNKS_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TURBOVEC_BIT_WIDTH, TURBOVEC_INDEX_PATH, TURBOVEC_METADATA_PATH

def chunk_text(text: str, source: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({"source": source, "text": chunk_text, "chunk_id": f"{source}#{start}"})
        if end == len(words):
            break
        start += size - overlap
    return chunks

def main():
    EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(EXTRACTS_DIR.glob("*.txt"))
    if not txt_files:
        print("No hay archivos extraídos. Ejecuta extract_pdfs.py primero.")
        return

    all_chunks = []
    for txt_path in tqdm(txt_files, desc="Chunkeando"):
        text = txt_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chunks = chunk_text(text, txt_path.stem)
        all_chunks.extend(chunks)

    chunks_json = CHUNKS_DIR / "chunks.json"
    chunks_json.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Chunks guardados: {len(all_chunks)} en {chunks_json}")

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in all_chunks]

    print("Generando embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"Construyendo índice TurboVec ({TURBOVEC_BIT_WIDTH}-bit, {embeddings.shape[1]} dim)...")
    index = IdMapIndex(dim=embeddings.shape[1], bit_width=TURBOVEC_BIT_WIDTH)
    ids = np.arange(len(embeddings), dtype=np.uint64)
    index.add_with_ids(embeddings, ids)

    index.write(str(TURBOVEC_INDEX_PATH))
    TURBOVEC_METADATA_PATH.write_text(
        json.dumps(metadatas, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nÍndice TurboVec guardado en {TURBOVEC_INDEX_PATH}")
    print(f"Metadatos guardados en {TURBOVEC_METADATA_PATH}")
    print(f"Total de chunks indexados: {len(ids)}")

if __name__ == "__main__":
    main()
