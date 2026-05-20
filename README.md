# rag-fsociety

RAG (Retrieval-Augmented Generation) con documentación de ciberseguridad para el modelo fsociety.

## Contenido

- `chat_rag.py` — Interfaz de chat con RAG vía Ollama
- `index_security_data.py` — Indexa datos de seguridad en ChromaDB

## Bases de conocimiento incluidas

- **how2heap**: 370+ exploits de heap en C (glibc 2.23 a 2.43)
- **CTF writeups**: 1,150+ writeups de 30+ competiciones
- **Ghidra**: Documentación y training del reverse engineering framework
- **JSONL**: 5 datasets de código seguro/vulnerable (pycode_vul, shellcode, securecode)
- **Phrack #61**: Magazine clásico de hacking
- **exploitdb-papers**: Papers de Exploit-DB

## Instalación

```bash
pip install -r requirements.txt
```

## Indexar datos

```bash
python index_security_data.py
```

## Ejecutar chat

```bash
python chat_rag.py
```

Requiere [Ollama](https://ollama.com) con el modelo `fsociety` o `murdokllmhack` como fallback.
