import os
import re
import time
import hashlib
from collections import deque
from getpass import getpass
from urllib.parse import urljoin, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# =========================
# Configurações
# =========================
BASE_URL = "https://arnaldojr.github.io/DisruptiveArchitectures/"
MAX_PAGES = 120
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MOSTRAR_TRECHOS = False
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL = "gemini-3.5-flash"

parsed_base = urlparse(BASE_URL)
BASE_HOST = parsed_base.netloc
BASE_PATH = parsed_base.path.rstrip("/") + "/"

SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".css", ".js", ".json", ".xml", ".zip", ".ipynb"
)

SYSTEM_INSTRUCTION = """
Você é um assistente acadêmico sobre a disciplina Disruptive Architectures.
Responda em português do Brasil, com clareza e objetividade.
Use exclusivamente o CONTEXTO fornecido. Não invente fatos, páginas ou datas.
Se o contexto não trouxer a resposta, diga explicitamente:
'Não encontrei essa informação na base consultada.'
Quando usar um trecho, cite-o no texto como [Fonte 1], [Fonte 2], etc.
Não confunda a base do site com informações externas ou conhecimento geral.
"""


def normalize_url(url):
    p = urlparse(url)
    path = p.path or "/"
    if path.endswith("/index.html"):
        path = path[:-len("index.html")]
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def is_allowed_url(url):
    p = urlparse(url)
    return (
        p.scheme in {"http", "https"}
        and p.netloc == BASE_HOST
        and (p.path == BASE_PATH.rstrip("/") or p.path.startswith(BASE_PATH))
        and not p.path.lower().endswith(SKIP_EXTENSIONS)
    )


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()

    for selector in [".md-sidebar", ".md-header", ".md-footer", ".md-nav", ".toc", ".table-of-contents"]:
        for tag in soup.select(selector):
            tag.decompose()

    content = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    title_tag = content.find("h1") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Sem título"

    text = content.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return title, text


def crawl_site(start_url, max_pages=MAX_PAGES):
    session = requests.Session()
    session.headers.update({"User-Agent": "RAG-educacional/1.0"})

    queue = deque([normalize_url(start_url)])
    visited = set()
    pages = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()

        if url in visited or not is_allowed_url(url):
            continue

        visited.add(url)

        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                continue

            raw_soup = BeautifulSoup(response.text, "html.parser")
            discovered_links = [a.get("href") for a in raw_soup.select("a[href]")]

            title, text = clean_html(response.text)

            if len(text) >= 120:
                pages.append({"url": url, "title": title, "text": text})

            for href in discovered_links:
                child = normalize_url(urljoin(url, href))
                if child not in visited and is_allowed_url(child):
                    queue.append(child)

        except requests.RequestException as exc:
            print(f"Aviso: falha ao acessar {url}: {exc}")

        time.sleep(0.03)

    return pages


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(pages):
    chunks = []

    for page in pages:
        page_chunks = split_into_chunks(page["text"])

        for position, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": hashlib.sha1(
                    f"{page['url']}#{position}".encode()
                ).hexdigest()[:12],
                "url": page["url"],
                "title": page["title"],
                "position": position,
                "text": chunk_text,
            })

    return pd.DataFrame(chunks).drop_duplicates(subset=["text"]).reset_index(drop=True)


def create_embeddings(chunks_df, embedding_model):
    chunk_texts = [
        f"Título: {row['title']}\nConteúdo: {row['text']}"
        for _, row in chunks_df.iterrows()
    ]

    return embedding_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

def retrieve(question, chunks_df, embeddings, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")

    scores = embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]

    results = chunks_df.iloc[indices].copy()
    results["score"] = scores[indices]

    return results.reset_index(drop=True)


def make_context(results):
    blocks = []

    for i, (_, row) in enumerate(results.iterrows(), start=1):
        blocks.append(
            f"[Fonte {i}]\n"
            f"Título: {row['title']}\n"
            f"URL: {row['url']}\n"
            f"Trecho: {row['text']}"
        )

    return "\n\n".join(blocks)


def generate_with_retry(client, prompt, max_retries=4):
    last_error = None

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            last_error = exc
            message = str(exc).upper()
            retryable = any(
                code in message
                for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"]
            )

            if not retryable or attempt == max_retries - 1:
                raise

            wait_seconds = min(30, 3 * (2 ** attempt))
            print(f"Serviço temporariamente indisponível. Nova tentativa em {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise last_error


def ask_rag(question, chunks_df, embeddings, embedding_model, client, top_k=TOP_K):
    results = retrieve(
        question,
        chunks_df,
        embeddings,
        embedding_model,
        top_k
    )

    context = make_context(results)

    prompt = f"""
{SYSTEM_INSTRUCTION}

CONTEXTO:
{context}

PERGUNTA DO USUÁRIO:
{question}
"""

    response = generate_with_retry(client, prompt)

    print("\nResposta:")
    print(response.text)

    print("\nFontes recuperadas:")

    for i, (_, item) in enumerate(results.iterrows(), start=1):
        print(
            f"[Fonte {i}] {item['title']} — {item['url']} "
            f"(score={item['score']:.3f})"
        )

        if MOSTRAR_TRECHOS:
            print(f"Trecho: {item['text'][:500]}...")
            print()

def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = getpass("Digite sua GOOGLE_API_KEY: ")

    return api_key


def main():
    print("=" * 70)
    print("RAG - Disruptive Architectures")
    print(f"Base consultada: {BASE_URL}")
    print("=" * 70)

    print("\n[1/3] Coletando páginas do site...")
    pages = crawl_site(BASE_URL)

    if not pages:
        raise RuntimeError("Nenhuma página foi coletada do site.")

    print(f"Páginas coletadas: {len(pages)}")

    print("\n[2/3] Criando chunks...")
    chunks_df = create_chunks(pages)
    print(f"Chunks criados: {len(chunks_df)}")

    print("\n[3/3] Criando embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = create_embeddings(chunks_df, embedding_model)
    print(f"Índice vetorial criado: {embeddings.shape}")

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    print("\nRAG pronto para perguntas!")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Pergunta: ").strip()

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o RAG.")
            break

        if not question:
            continue

        try:
            ask_rag(
                question,
                chunks_df,
                embeddings,
                embedding_model,
                client,
            )
        except Exception as exc:
            print(f"\nErro ao gerar resposta: {type(exc).__name__}: {exc}")

        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
