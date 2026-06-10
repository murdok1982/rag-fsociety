# rag-fsociety

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/ChromaDB-✓-brightgreen" alt="ChromaDB"/>
</p>

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

## 🤖 Recomendación

Para tareas de ciberseguridad, exploiting y reversing, este RAG está diseñado para funcionar con [fsociety](https://huggingface.co/murdok1982/fsociety), un modelo fine-tuned sobre Qwen2.5-Coder-1.5B-Instruct con 169K ejemplos de seguridad. Corre 100% local con Ollama:

```bash
ollama pull murdok1982/fsociety
```

---

## Support / Apoya este proyecto

I build open-source projects focused on applied AI, automation, and data intelligence.
Over on my GitHub you'll find things like AI-powered analysis engines, OSINT platforms for open-source research, Windows automation tools, and experiments with language models.
Everything is public and free, so anyone can use it, study it, or build on top of it. github.com/murdok1982

Keeping these projects alive takes a lot of hours. If any of them have helped you out or you just like what I'm doing, you can support me with a coffee: ko-fi.com/murdok1982

Every contribution goes straight back into shipping more open-source code.
