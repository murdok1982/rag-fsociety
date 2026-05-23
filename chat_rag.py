import sys, json, urllib.request
import hispan_shield_guardian  # noqa: F401
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from sentence_transformers import SentenceTransformer
from config import VECTOR_DB_DIR, EMBEDDING_MODEL

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "fsociety"
FALLBACK_MODEL = "murdokllmhack"
MEMORY_FILE = Path(__file__).parent / "memoria.json"

print("Cargando base vectorial de seguridad...")
if VECTOR_DB_DIR.exists():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        vector_db = chroma_client.get_collection("seguridad")
        print(f"  Coleccion 'seguridad' lista ({vector_db.count()} chunks)")
    except:
        vector_db = None
        print("  Coleccion 'seguridad' no encontrada")
else:
    embedder = None
    vector_db = None
    print("  Base vectorial no existe")

if MEMORY_FILE.exists():
    memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
else:
    memory = []
    MEMORY_FILE.write_text("[]", encoding="utf-8")

def query_rag(query: str, top_k: int = 5) -> str:
    if not vector_db:
        return ""
    q_emb = embedder.encode([query])[0]
    results = vector_db.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    return "\n---\n".join(results["documents"][0]) if results["documents"][0] else ""

def ask_ollama(messages: list, model: str = None) -> str:
    m = model or MODEL_NAME
    data = json.dumps({"model": m, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["message"]["content"]
    except Exception as e:
        if model is None and FALLBACK_MODEL != MODEL_NAME:
            print(f"  {MODEL_NAME} no disponible, usando {FALLBACK_MODEL}")
            return ask_ollama(messages, model=FALLBACK_MODEL)
        return f"[Error conectando a Ollama: {e}]"

print(f"\nfsociety + RAG seguridad listo! ('salir' para salir, 'reset' borra memoria)")
print(f"Modelo: {MODEL_NAME} | Fallback: {FALLBACK_MODEL}\n")

history = []

SYSTEM_PROMPT = """Eres fsociety, un experto en ciberseguridad, reversing, exploiting y hacking etico.
Tienes acceso a una base de conocimiento con:
- CTF writeups de mas de 30 competiciones
- Tecnicas de heap exploitation (how2heap)
- Documentacion de Ghidra (reversing)
- Papers de exploitdb
- Datasets de codigo seguro y vulnerable
- Phrack magazine
Usa este conocimiento para dar respuestas precisas y tecnicas.
Siempre prioriza la seguridad y el hacking etico."""

while True:
    user_input = input("Tu: ")
    if user_input.lower() in ("salir", "exit", "quit"):
        break
    if user_input.lower() == "reset":
        memory = []
        history = []
        MEMORY_FILE.write_text("[]", encoding="utf-8")
        print("Memoria borrada.\n")
        continue

    rag_context = query_rag(user_input)
    if rag_context:
        print("  (usando RAG + base de seguridad)")

    context = f"\n\nInformacion relevante:\n{rag_context}" if rag_context else ""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
    for m in history[-10:]:
        messages.append(m)
    messages.append({"role": "user", "content": user_input})

    response = ask_ollama(messages)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    memory.append({"user": user_input, "assistant": response})
    MEMORY_FILE.write_text(json.dumps(memory[-100:], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fsociety: {response}\n")
