import json, re, random
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import CHUNKS_DIR, DATASET_DIR

random.seed(42)

SYSTEM_PROMPT = "Eres un experto en ciberseguridad, redes y hacking etico. Responde preguntas tecnicas con precision basandote en el conocimiento proporcionado."

QUESTION_TEMPLATES = [
    "Explica el concepto principal de este fragmento: {text}",
    "Define los terminos clave explicados en: {text}",
    "Resume en una oracion la idea central de: {text}",
    "Que informacion relevante se extrae de este texto? {text}",
    "Cual es la importancia de lo descrito en: {text}",
    "Describe el proceso o mecanismo explicado en: {text}",
    "Cuales son las implicaciones de seguridad de: {text}",
    "Como se aplica en ciberseguridad lo siguiente? {text}",
    "Enumera los puntos clave del siguiente fragmento: {text}",
    "Que relacion tiene este concepto con la seguridad informatica? {text}",
    "Compara este concepto con alternativas similares: {text}",
    "Por que es relevante en hacking etico lo siguiente? {text}",
    "Cual es la vulnerabilidad o riesgo descrito en: {text}",
    "Que medidas de mitigacion se mencionan en: {text}",
    "Como implementarias tecnicamente lo descrito en: {text}",
]

SHORT_QS = [
    "Explica que significa.",
    "Define los terminos clave.",
    "Cual es la idea principal?",
    "Que relevancia tiene en ciberseguridad?",
    "Como se aplica en la practica?",
    "Cuales son los riesgos asociados?",
    "Enumera los puntos clave.",
    "Por que es importante?",
]

def extract_terms(text):
    words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', '', text).split() if len(w) > 4 and w[0].isupper()]
    return list(set(w for w in words if w.lower() not in {"this", "that", "with", "from", "which", "what", "when", "where", "there", "their", "about", "would", "could", "should", "these", "those", "while", "after", "before", "other", "first", "second", "third"}))

def generate_qa_pairs(chunk, source):
    text = chunk.replace("\n", " ")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 60]
    qa_pairs = []
    chosen = random.sample(sentences, min(5, len(sentences))) if len(sentences) > 2 else sentences

    for sent in chosen[:4]:
        if len(sent.split()) < 8:
            continue

        template = random.choice(QUESTION_TEMPLATES)
        question = template.format(text=sent[:300])
        short_q = random.choice(SHORT_QS)
        full_question = f"{question}\n\n{short_q}"

        terms = extract_terms(sent)
        if terms and random.random() < 0.3:
            key_term = random.choice(terms)
            full_question = f"Explica el concepto de '{key_term}' en el contexto de: {sent[:300]}"

        response = sent[:500]

        if random.random() < 0.3:
            words = response.split()
            mid = len(words) // 2
            response = " ".join(words[:mid]) + "\n\nPuntos clave:\n- " + "\n- ".join(words[mid:mid+3]) if len(words) > 6 else response

        qa_pairs.append({
            "instruction": "Responde la siguiente pregunta sobre " + source.replace("_", " ").replace("/", " "),
            "input": full_question,
            "output": response,
            "source": source,
        })

    return qa_pairs

def format_chat(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example['output']},
    ]
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}\n<|im_end|>\n"
    return {"text": text, "messages": messages}

def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    chunks_file = CHUNKS_DIR / "chunks.json"
    if not chunks_file.exists():
        print("No se encuentra chunks.json."); return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    print(f"Generando dataset desde {len(chunks)} chunks...")

    all_qa = []
    for c in tqdm(chunks, desc="Generando Q&A"):
        all_qa.extend(generate_qa_pairs(c["text"], c["source"]))

    random.shuffle(all_qa)
    split = int(len(all_qa) * 0.9)
    train_data = all_qa[:split]
    val_data = all_qa[split:]

    train_formatted = [format_chat(ex) for ex in tqdm(train_data, desc="Formateando train")]
    val_formatted = [format_chat(ex) for ex in tqdm(val_data, desc="Formateando val")]

    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_formatted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for ex in val_formatted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    for hf_dir in [DATASET_DIR / "hf_dataset"]:
        hf_dir.mkdir(exist_ok=True)
        hf_train = hf_dir / "train.jsonl"
        hf_val = hf_dir / "validation.jsonl"
        with open(hf_train, "w", encoding="utf-8") as f:
            for ex in train_formatted:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
        with open(hf_val, "w", encoding="utf-8") as f:
            for ex in val_formatted:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    print(f"\nDataset generado:")
    print(f"  Train: {len(train_formatted)} ejemplos -> {train_file}")
    print(f"  Val:   {len(val_formatted)} ejemplos -> {val_file}")

if __name__ == "__main__":
    main()
