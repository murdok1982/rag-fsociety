import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import CHUNKS_DIR, DATASET_DIR

EXCLUDE_SOURCES = {
    "modernC", "modern_C", "Ser Digital", "ser_digital",
    "Hacia una reflexion historica de las TIC", "reflexion_historica",
    "memoria",
}

MIN_SENTENCE_WORDS = 12
MIN_SENTENCE_CHARS = 60

def is_valid_sentence(s):
    s = s.strip()
    if len(s) < MIN_SENTENCE_CHARS or len(s.split()) < MIN_SENTENCE_WORDS:
        return False
    if s.lower().startswith("machine translated by google"):
        return False
    if s.lower().startswith("machine translation"):
        return False
    if re.search(r'[^\x00-\x7F]', s) and len(re.findall(r'[^\x00-\x7F]', s)) > len(s) * 0.3:
        return False
    if s.endswith((",", ":", ";", "de", "del", "la", "el", "un", "una")):
        return False
    return True

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    for tag in ["<|im_start|>", "<|im_end|>", "<|im_sep|>"]:
        text = text.replace(tag, "")
    return text.strip()

def deduplicate(entries, key_fn):
    seen = set()
    unique = []
    for e in entries:
        k = key_fn(e)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique

def main():
    chunks_file = CHUNKS_DIR / "chunks.json"
    if not chunks_file.exists():
        print("No hay chunks.json. Ejecuta build_vector_db.py primero.")
        sys.exit(1)

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    total_before = len(chunks)
    print(f"Total chunks antes: {total_before}")

    filtered = [c for c in chunks if not any(ex in c["source"] for ex in EXCLUDE_SOURCES)]
    print(f"Tras excluir fuentes no-seguridad: {len(filtered)} (eliminados {total_before - len(filtered)})")

    filtered = deduplicate(filtered, lambda c: c["text"][:200])
    print(f"Tras deduplicar: {len(filtered)} (eliminados {len(chunks) - len(filtered)})")

    for c in filtered:
        c["text"] = clean_text(c["text"])

    json.dump(filtered, chunks_file.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Chunks limpios guardados en {chunks_file}")

if __name__ == "__main__":
    main()
