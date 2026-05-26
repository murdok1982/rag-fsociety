import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import IdMapIndex
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import EMBEDDING_MODEL, TURBOVEC_INDEX_PATH, TURBOVEC_METADATA_PATH

def query_vector_db(query: str, top_k: int = 5):
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = IdMapIndex.load(str(TURBOVEC_INDEX_PATH))
    metadata = json.loads(TURBOVEC_METADATA_PATH.read_text(encoding="utf-8"))

    q_emb = model.encode([query]).reshape(1, -1)
    scores, ids = index.search(q_emb, k=top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        i = int(idx)
        if i < len(metadata):
            results.append({
                "source": metadata[i]["source"],
                "chunk_id": metadata[i]["chunk_id"],
                "text": metadata[i]["text"],
                "score": float(score),
            })
    return results

def main():
    parser = argparse.ArgumentParser(description="Consulta la base vectorial (TurboVec)")
    parser.add_argument("query", type=str, help="Texto de consulta")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados")
    args = parser.parse_args()

    results = query_vector_db(args.query, args.top_k)
    print(f"\nConsulta: {args.query}\n")
    print("Resultados:\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] Fuente: {r['source']} (score: {r['score']:.4f})")
        print(f"    {r['text'][:200]}...")
        print()

if __name__ == "__main__":
    main()
