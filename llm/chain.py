from typing import Dict, List

import requests
from google import generativeai as google_genai
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = (
    "Responda apenas com base no contexto fornecido. "
    "Se a resposta nao estiver no contexto, diga: "
    "'Nao encontrei essa informacao no documento.'"
)


def _normalize_embedding(values: List[float], target_dim: int = 1536) -> List[float]:
    if len(values) == target_dim:
        return values

    if len(values) > target_dim:
        return values[:target_dim]

    padded = list(values)
    padded.extend([0.0] * (target_dim - len(values)))
    return padded


def _openai_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [item.embedding for item in response.data]
    return [_normalize_embedding(v) for v in vectors]


def _google_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    google_genai.configure(api_key=api_key)
    vectors: List[List[float]] = []
    for text in texts:
        response = google_genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        vectors.append(_normalize_embedding(response["embedding"]))
    return vectors


def _huggingface_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    vectors: List[List[float]] = []
    headers = {"Authorization": f"Bearer {api_key}"}

    for text in texts:
        resp = requests.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json={"inputs": text},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()

        if not payload or not isinstance(payload, list):
            raise RuntimeError("Resposta invalida da API Hugging Face para embeddings")

        if isinstance(payload[0], list):
            token_vectors = payload
            mean_vector = [sum(col) / len(col) for col in zip(*token_vectors)]
            vectors.append(_normalize_embedding(mean_vector))
        else:
            vectors.append(_normalize_embedding(payload))

    return vectors


def get_embeddings(texts: List[str], provider: str, api_keys: Dict[str, str]) -> List[List[float]]:
    if not texts:
        return []

    if provider == "openai":
        return _openai_embeddings(texts, api_keys["openai"])
    if provider == "google":
        return _google_embeddings(texts, api_keys["google"])
    if provider == "huggingface":
        return _huggingface_embeddings(texts, api_keys["huggingface"])

    raise ValueError(f"Provedor nao suportado: {provider}")


def embed_query(query: str, provider: str, api_keys: Dict[str, str]) -> List[float]:
    vectors = get_embeddings(texts=[query], provider=provider, api_keys=api_keys)
    if not vectors:
        raise RuntimeError("Falha ao gerar embedding da pergunta")
    return vectors[0]


def _openai_answer(question: str, context: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _google_answer(question: str, context: str, api_key: str) -> str:
    google_genai.configure(api_key=api_key)
    model = google_genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def _huggingface_answer(question: str, context: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )

    resp = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json={"inputs": prompt},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, list) and payload and "generated_text" in payload[0]:
        return payload[0]["generated_text"].strip()

    raise RuntimeError("Resposta invalida da API Hugging Face para geracao")


def generate_answer(question: str, context: str, provider: str, api_keys: Dict[str, str]) -> str:
    if provider == "openai":
        return _openai_answer(question, context, api_keys["openai"])
    if provider == "google":
        return _google_answer(question, context, api_keys["google"])
    if provider == "huggingface":
        return _huggingface_answer(question, context, api_keys["huggingface"])

    raise ValueError(f"Provedor nao suportado: {provider}")
