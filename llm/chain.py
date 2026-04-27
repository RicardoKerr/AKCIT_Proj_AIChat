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


def _openai_embeddings(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return [_normalize_embedding(v) for v in vectors]


def _google_embeddings(texts: List[str], api_key: str, model: str = "models/text-embedding-004") -> List[List[float]]:
    google_genai.configure(api_key=api_key)
    vectors: List[List[float]] = []
    for text in texts:
        response = google_genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document",
        )
        vectors.append(_normalize_embedding(response["embedding"]))
    return vectors


def _huggingface_embeddings(
    texts: List[str],
    api_key: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[List[float]]:
    vectors: List[List[float]] = []
    headers = {"Authorization": f"Bearer {api_key}"}

    for text in texts:
        resp = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
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


def get_embeddings(
    texts: List[str],
    provider: str,
    api_keys: Dict[str, str],
    embedding_model: str = "",
) -> List[List[float]]:
    if not texts:
        return []

    if provider == "openai":
        model = embedding_model or "text-embedding-3-small"
        if "embedding" not in model:
            raise ValueError(
                f"Modelo invalido para embeddings OpenAI: '{model}'. "
                "Use um modelo de embeddings, como text-embedding-3-small."
            )
        return _openai_embeddings(texts, api_keys["openai"], model=model)
    if provider == "google":
        model = embedding_model or "models/text-embedding-004"
        if "embed" not in model:
            raise ValueError(
                f"Modelo invalido para embeddings Google: '{model}'. "
                "Use um modelo de embeddings, como models/text-embedding-004."
            )
        return _google_embeddings(texts, api_keys["google"], model=model)
    if provider == "huggingface":
        model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        return _huggingface_embeddings(texts, api_keys["huggingface"], model=model)

    raise ValueError(f"Provedor nao suportado: {provider}")


def embed_query(
    query: str,
    provider: str,
    api_keys: Dict[str, str],
    embedding_model: str = "",
) -> List[float]:
    vectors = get_embeddings(
        texts=[query],
        provider=provider,
        api_keys=api_keys,
        embedding_model=embedding_model,
    )
    if not vectors:
        raise RuntimeError("Falha ao gerar embedding da pergunta")
    return vectors[0]


def _openai_answer(question: str, context: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _google_answer(question: str, context: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    google_genai.configure(api_key=api_key)
    model = google_genai.GenerativeModel(model_name=model_name)
    prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def _huggingface_answer(question: str, context: str, api_key: str, model: str = "google/flan-t5-large") -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n"
    )

    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json={"inputs": prompt},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, list) and payload and "generated_text" in payload[0]:
        return payload[0]["generated_text"].strip()

    raise RuntimeError("Resposta invalida da API Hugging Face para geracao")


def generate_answer(
    question: str,
    context: str,
    provider: str,
    api_keys: Dict[str, str],
    chat_model: str = "",
) -> str:
    if provider == "openai":
        model = chat_model or "gpt-4o-mini"
        return _openai_answer(question, context, api_keys["openai"], model=model)
    if provider == "google":
        model = chat_model or "gemini-1.5-flash"
        return _google_answer(question, context, api_keys["google"], model_name=model)
    if provider == "huggingface":
        model = chat_model or "google/flan-t5-large"
        return _huggingface_answer(question, context, api_keys["huggingface"], model=model)

    raise ValueError(f"Provedor nao suportado: {provider}")


def list_available_models(provider: str, api_key: str, limit: int = 20) -> List[str]:
    if not api_key:
        raise ValueError("API key nao informada")

    if provider == "openai":
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        names = sorted([model.id for model in models.data])
        return names[:limit]

    if provider == "google":
        google_genai.configure(api_key=api_key)
        names: List[str] = []
        for model in google_genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods or "embedContent" in methods:
                names.append(model.name)
        names = sorted(names)
        return names[:limit]

    if provider == "huggingface":
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            "https://huggingface.co/api/models?limit=100&sort=downloads&direction=-1",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            raise RuntimeError("Resposta invalida ao listar modelos do Hugging Face")

        names: List[str] = []
        for item in payload:
            model_id = item.get("id")
            if model_id:
                names.append(model_id)
            if len(names) >= limit:
                break
        return names

    raise ValueError(f"Provedor nao suportado: {provider}")
