#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def api(path, data):
    r = urllib.request.Request(
        "http://localhost:11434" + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=600) as x:
        return json.load(x)


def emb(t):
    return api("/api/embed", {"model": "nomic-embed-text:latest", "input": t})[
        "embeddings"
    ][0]


def sim(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def text(p):
    if p.suffix == ".txt":
        return p.read_text()
    r = subprocess.run(
        ["pdftotext", "-layout", str(p), "-"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if r.returncode:
        raise RuntimeError("No se pudo leer el PDF: " + r.stderr.strip())
    if len(r.stdout.strip()) >= 80:
        return r.stdout
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "El PDF parece escaneado: instala tesseract-ocr y tesseract-ocr-spa."
        )
    with tempfile.TemporaryDirectory() as d:
        prefix = str(Path(d) / "page")
        pages = subprocess.run(
            ["pdftoppm", "-r", "200", "-png", str(p), prefix],
            stderr=subprocess.PIPE,
            text=True,
        )
        if pages.returncode:
            raise RuntimeError(
                "No se pudieron convertir las páginas para OCR: " + pages.stderr.strip()
            )
        images = sorted(Path(d).glob("page-*.png"))
        if not images:
            raise RuntimeError("El PDF no produjo páginas para OCR.")
        return "\n".join(
            subprocess.run(
                ["tesseract", str(i), "stdout", "-l", "spa+eng"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            ).stdout
            for i in images
        )


def index(folder, out):
    rows = []
    for p in Path(folder).glob("*"):
        if p.suffix not in (".txt", ".pdf"):
            continue
        t = text(p)
        for n, i in enumerate(range(0, len(t), 750), 1):
            chunk = t[i : i + 900]
            rows.append(
                {"file": p.name, "chunk": n, "text": chunk, "embedding": emb(chunk)}
            )
    Path(out).write_text(json.dumps(rows, ensure_ascii=False))
    print(len(rows), "fragmentos indexados")


def ask(db, q, model):
    rows = json.loads(Path(db).read_text())
    v = emb(q)
    hits = sorted(rows, key=lambda x: sim(v, x["embedding"]), reverse=True)[:4]
    evidence = "\n\n".join(
        f'[{x["file"]}, fragmento {x["chunk"]}]\n{x["text"]}' for x in hits
    )
    prompt = (
        "Responde en español solo con evidencia; cita [archivo, fragmento]. Si no basta, dilo.\n"
        + evidence
        + "\nPregunta: "
        + q
    )
    print(
        api(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
            },
        )["message"]["content"]
    )


def main():
    parser = argparse.ArgumentParser(description="Agente documental local (RAG)")
    commands = parser.add_subparsers(dest="cmd", required=True)

    index_cmd = commands.add_parser("index", help="indexar una carpeta de .txt y .pdf")
    index_cmd.add_argument("folder")
    index_cmd.add_argument("--out", default="indice.json")

    ask_cmd = commands.add_parser("ask", help="preguntar sobre un índice")
    ask_cmd.add_argument("index")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--model", default="qwen3:14b")

    args = parser.parse_args()
    if args.cmd == "index":
        index(args.folder, args.out)
    else:
        ask(args.index, args.question, args.model)


if __name__ == "__main__":
    main()
