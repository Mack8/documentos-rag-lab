# Local document agent lab / Laboratorio de agente documental local

**English:** a small, dependency-free RAG (retrieval-augmented generation) lab that runs entirely on your machine. It extracts text from `.txt` and PDF files (falling back to Tesseract OCR for scanned PDFs), builds embeddings with `nomic-embed-text` through [Ollama](https://ollama.com), retrieves the closest chunks, and asks a local model to answer **with source citations**. It is a teaching project: it does not train a model and nothing leaves your computer. The detailed documentation below is in Spanish.

```bash
python3 document_agent.py index ./my-docs --out index.json
python3 document_agent.py ask index.json "What is the cancellation notice period?"
```

---

Este proyecto enseña RAG: extraer texto, crear embeddings, recuperar evidencia y pedir una respuesta con citas. No entrena un modelo ni envía documentos fuera del equipo.

```text
documentos -> fragmentos -> nomic-embed-text -> índice -> búsqueda -> modelo -> respuesta + fuentes
```

## Probar con texto seguro

```bash
mkdir -p /tmp/documentos-demo
printf 'El contrato permite cancelar con aviso de 30 días.\n' > /tmp/documentos-demo/contrato.txt
python3 document_agent.py index /tmp/documentos-demo --out /tmp/indice-demo.json
python3 document_agent.py ask /tmp/indice-demo.json '¿Cuál es el plazo de cancelación?'
```

Para PDFs digitales se usa `pdftotext`. Si el PDF no contiene texto (por ejemplo, es un escaneo), el agente activa OCR automáticamente usando Tesseract. Instálalo una vez:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
```

El reconocimiento usará español e inglés. OCR puede cometer errores, especialmente en fotos borrosas, tablas, sellos o escritura a mano; las respuestas del agente deben conservar sus citas para poder verificarlas contra la página original.

## Uso empresarial

Separar índices por empresa/tenant, conservar archivo y página de cada fuente, exigir citas, controlar quién puede indexar/consultar y registrar accesos.
