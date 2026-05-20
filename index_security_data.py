import sys, json, re, tarfile, io
from pathlib import Path
sys.path.insert(0, r"C:\Users\USUARIO\Desktop\Formacion\Entrenamiento")
from config import VECTOR_DB_DIR, EMBEDDING_MODEL

SECURITY_DIR = Path(__file__).parent / "security_data"
EXCLUDE_DIRS = {".git", "__pycache__", ".github", "ci", "gradle", "build", "lib", "licenses"}
EXCLUDE_EXTS = {".png", ".jpg", ".gif", ".svg", ".bmp", ".exe", ".bin", ".so", ".o", ".class",
                ".jar", ".war", ".zip", ".gz", ".bz2", ".rar", ".7z", ".pcap", ".pcapng", ".wav",
                ".ttf", ".woff", ".eot"}

print(f"Base vectorial: {VECTOR_DB_DIR}")
print(f"Escaneando: {SECURITY_DIR}")

docs = []

# ── JSONL files ──
for f in SECURITY_DIR.glob("*.jsonl"):
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                data = json.loads(line)
                text = data.get("text", "") or data.get("messages", "") or str(data)
                if isinstance(text, list):
                    text = " ".join(m.get("content", "") for m in text if "content" in m)
                docs.append({"text": text[:2000], "source": f.name})
            except json.JSONDecodeError:
                pass
    print(f"  {f.name}: indexado")

# ── phrack61.tar.gz ──
phrack = SECURITY_DIR / "phrack61.tar.gz"
if phrack.exists():
    with tarfile.open(phrack, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and Path(m.name).suffix in (".txt", ".md", ""):
                content = tar.extractfile(m).read().decode("utf-8", errors="ignore")
                docs.append({"text": content[:2000], "source": f"phrack/{m.name}"})
    print("  phrack61: indexado")

# ── Recorrer subdirectorios ──
for folder in SECURITY_DIR.iterdir():
    if not folder.is_dir() or folder.name in EXCLUDE_DIRS:
        continue

    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.suffix in EXCLUDE_EXTS:
            continue
        if any(p.name in EXCLUDE_DIRS for p in f.parents):
            continue
        if f.suffix not in (".c", ".py", ".md", ".txt", ".html",
                            ".php", ".js", ".rb", ".go", ".rs", ".sh",
                            ".yaml", ".yml", ".cfg", ".conf", ".xml"):
            continue
        # Skip Ghidra Java source (only keep docs/training)
        if "ghidra" in f.parts and f.suffix == ".java":
            continue

        try:
            text = f.read_text("utf-8", errors="ignore")
        except:
            continue

        # Skip large files
        if len(text) > 100000:
            text = text[:100000]

        rel = f.relative_to(folder)
        docs.append({"text": text, "source": f"{folder.name}/{rel}"})

    print(f"  {folder.name}: {len([d for d in docs if d['source'].startswith(folder.name)])} docs")

print(f"\nTotal documentos: {len(docs)}")

# ── Chunking ──
def chunk_text(text, size=512, overlap=64):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

chunks = []
for d in docs:
    for chunk in chunk_text(d["text"]):
        chunks.append({"text": chunk, "source": d["source"]})

print(f"Total chunks: {len(chunks)}")

# ── Embeddings ──
print("Generando embeddings...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDING_MODEL)

import chromadb
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

try:
    client.delete_collection("seguridad")
except:
    pass

collection = client.create_collection(
    name="seguridad",
    metadata={"hnsw:space": "cosine"}
)

BATCH = 100
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i+BATCH]
    texts = [c["text"] for c in batch]
    ids = [f"doc_{i+j}" for j in range(len(batch))]
    metas = [{"source": c["source"]} for c in batch]
    embs = embedder.encode(texts).tolist()
    collection.add(embeddings=embs, documents=texts, metadatas=metas, ids=ids)

print(f"Indexacion completa. Coleccion 'seguridad' con {len(chunks)} chunks")
print(f"Base vectorial en: {VECTOR_DB_DIR}")
