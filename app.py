import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from ingestion.loader import chunk_text, extract_text_from_uploaded_file
from llm.chain import embed_query, generate_answer, get_embeddings, list_available_models
from retriever.chroma_store import ChromaVectorStore


load_dotenv()


def _ensure_session_defaults() -> None:
    defaults = {
        "index_ready": False,
        "indexed_docs": [],
        "indexed_provider": "",
        "indexed_collection": "rag_chunks",
        "suggested_questions": [],
        "available_models": [],
        "available_models_provider": "",
        "selected_chat_model": "",
        "selected_embedding_model": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _validate_runtime(provider: str, api_key: str, chat_model: str, embedding_model: str) -> bool:
    if not api_key:
        st.error(f"Informe a chave do provedor selecionado: {provider}.")
        return False

    if not embedding_model:
        st.error("Selecione um modelo de embeddings para processar e consultar documentos.")
        return False

    if not chat_model:
        st.error("Selecione um modelo de resposta para gerar as respostas do chat.")
        return False

    return True


def _filter_models_for_tasks(provider: str, models: List[str]) -> Tuple[List[str], List[str]]:
    embedding_keywords = ["embed", "embedding", "text-embedding", "e5", "bge", "minilm", "gte"]

    if provider == "openai":
        embedding_models = [m for m in models if "embedding" in m]
        chat_models = [
            m for m in models
            if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
        ]
    elif provider == "google":
        embedding_models = [m for m in models if "embed" in m]
        chat_models = [m for m in models if "gemini" in m and "embed" not in m]
    else:
        embedding_models = [m for m in models if any(keyword in m.lower() for keyword in embedding_keywords)]
        chat_models = [m for m in models if m not in embedding_models]

    if not embedding_models:
        # Evita fallback para modelos de chat no caso OpenAI/Google.
        if provider == "openai":
            embedding_models = ["text-embedding-3-small"]
        elif provider == "google":
            embedding_models = ["models/text-embedding-004"]
        else:
            embedding_models = models[:]
    if not chat_models:
        chat_models = models[:]

    return embedding_models, chat_models


def _normalize_token(token: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9à-úÀ-Ú_-]", "", token.lower()).strip("_")
    return cleaned


def _suggest_questions_from_chunks(chunks: List[dict], limit: int = 3) -> List[str]:
    stopwords = {
        "a", "o", "os", "as", "de", "da", "do", "das", "dos", "e", "em", "para", "por",
        "que", "com", "na", "no", "nas", "nos", "um", "uma", "uns", "umas", "se", "ou",
        "ao", "aos", "sua", "seu", "suas", "seus", "como", "ser", "sao", "são", "mais", "menos",
        "sobre", "pelo", "pela", "pelos", "pelas", "isso", "essa", "esse", "este", "esta", "estes",
        "estas", "tambem", "também", "entre", "quando", "onde", "qual", "quais", "porque", "pois",
    }

    token_counter: Counter = Counter()
    for chunk in chunks[:6]:
        for raw in chunk.get("content", "").split():
            token = _normalize_token(raw)
            if len(token) < 4 or token.isdigit() or token in stopwords:
                continue
            token_counter[token] += 1

    keywords = [token for token, _ in token_counter.most_common(8)]
    questions: List[str] = []

    for keyword in keywords:
        candidate = f"O que o documento diz sobre {keyword}?"
        if candidate not in questions:
            questions.append(candidate)
        if len(questions) >= limit:
            return questions

    fallback_questions = [
        "Qual e o objetivo principal deste documento?",
        "Quais pontos mais importantes o documento apresenta?",
        "Quais recomendacoes praticas o documento sugere?",
    ]
    for fallback in fallback_questions:
        if fallback not in questions:
            questions.append(fallback)
        if len(questions) >= limit:
            break

    return questions[:limit]


st.set_page_config(page_title="Chatbot RAG", page_icon="RAG", layout="wide")
_ensure_session_defaults()

st.title("Chatbot de Perguntas e Respostas (RAG)")
st.caption("Responda perguntas com base nos documentos enviados.")

with st.sidebar:
    st.header("Configuracao do LLM")
    st.caption("Fluxo: provedor -> API key -> modelos -> processamento")

    previous_provider = st.session_state.get("current_provider", "openai")
    provider = st.selectbox(
        "Provedor de IA",
        options=["openai", "google", "huggingface"],
        index=0,
    )
    st.session_state.current_provider = provider

    if previous_provider != provider:
        st.session_state.available_models = []
        st.session_state.available_models_provider = ""
        st.session_state.selected_chat_model = ""
        st.session_state.selected_embedding_model = ""

    st.subheader("1) API key")
    api_keys: Dict[str, str] = {
        "openai": st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
        "google": st.session_state.get("google_api_key", os.getenv("GOOGLE_API_KEY", "")),
        "huggingface": st.session_state.get("hf_api_key", os.getenv("HUGGINGFACE_API_TOKEN", "")),
    }

    key_labels = {
        "openai": "OpenAI API Key",
        "google": "Google API Key",
        "huggingface": "Hugging Face API Token",
    }
    key_state_names = {
        "openai": "openai_api_key",
        "google": "google_api_key",
        "huggingface": "hf_api_key",
    }

    api_keys[provider] = st.text_input(
        key_labels[provider],
        type="password",
        value=api_keys[provider],
        key=key_state_names[provider],
    )

    st.subheader("2) Modelos")
    if st.button("Conectar API e carregar modelos"):
        selected_key = api_keys.get(provider, "")
        if not selected_key:
            st.error("Informe a API key do provedor selecionado para carregar os modelos.")
        else:
            try:
                models = list_available_models(provider=provider, api_key=selected_key, limit=40)
                if not models:
                    st.error("Nenhum modelo retornado para esta API key.")
                else:
                    embedding_models, chat_models = _filter_models_for_tasks(provider=provider, models=models)
                    st.session_state.available_models = models
                    st.session_state.available_models_provider = provider
                    st.session_state.selected_embedding_model = embedding_models[0]
                    st.session_state.selected_chat_model = chat_models[0]
                    st.success(f"{len(models)} modelos carregados para {provider}.")
            except Exception as exc:
                st.session_state.available_models = []
                st.session_state.available_models_provider = ""
                st.session_state.selected_chat_model = ""
                st.session_state.selected_embedding_model = ""
                st.error(f"Falha ao listar modelos: {exc}")

    model_source_ok = (
        st.session_state.available_models
        and st.session_state.available_models_provider == provider
    )

    selected_chat_model = ""
    selected_embedding_model = ""
    if model_source_ok:
        embedding_options, chat_options = _filter_models_for_tasks(
            provider=provider,
            models=st.session_state.available_models,
        )

        if provider == "openai" and all("embedding" not in m for m in embedding_options):
            st.warning(
                "Nenhum modelo de embeddings foi retornado pela API para OpenAI. "
                "Usando fallback: text-embedding-3-small."
            )

        if provider == "google" and all("embed" not in m for m in embedding_options):
            st.warning(
                "Nenhum modelo de embeddings foi retornado pela API para Google. "
                "Usando fallback: models/text-embedding-004."
            )

        if st.session_state.selected_embedding_model not in embedding_options:
            st.session_state.selected_embedding_model = embedding_options[0]
        if st.session_state.selected_chat_model not in chat_options:
            st.session_state.selected_chat_model = chat_options[0]

        selected_embedding_model = st.selectbox(
            "Modelo de embeddings",
            options=embedding_options,
            index=embedding_options.index(st.session_state.selected_embedding_model),
        )
        selected_chat_model = st.selectbox(
            "Modelo de resposta",
            options=chat_options,
            index=chat_options.index(st.session_state.selected_chat_model),
        )

        st.session_state.selected_embedding_model = selected_embedding_model
        st.session_state.selected_chat_model = selected_chat_model

        with st.expander("Ver modelos carregados"):
            for idx, model_name in enumerate(st.session_state.available_models[:30], start=1):
                st.caption(f"{idx}. {model_name}")
    else:
        st.info("Conecte a API key para carregar e selecionar os modelos.")

    st.subheader("3) Recuperacao")
    top_k = st.slider("Top-K", min_value=3, max_value=5, value=4)
    similarity_threshold = st.slider("Threshold de similaridade", 0.0, 1.0, 0.25, 0.05)
    diagnostic_mode = st.checkbox(
        "Modo diagnostico",
        value=False,
        help="Mostra scores de similaridade e chunks retornados, incluindo os descartados pelo threshold.",
    )

    if st.button("Limpar indice atual"):
        st.session_state.index_ready = False
        st.session_state.indexed_docs = []
        st.session_state.indexed_provider = ""
        st.session_state.suggested_questions = []
        st.session_state.indexed_collection = f"rag_chunks_{provider}"
        st.success("Indice removido da sessao. Processe os documentos novamente.")

st.divider()

uploaded_files = st.file_uploader(
    "Upload de documentos",
    type=["pdf", "txt", "docx", "xlsx", "csv"],
    accept_multiple_files=True,
)
active_api_key = api_keys.get(provider, "")

if st.session_state.index_ready:
    indexed_files_text = ", ".join(st.session_state.indexed_docs) if st.session_state.indexed_docs else "nao informado"
    st.info(
        "Indice ativo na sessao. "
        f"Provedor: {st.session_state.indexed_provider}. "
        f"Colecao: {st.session_state.indexed_collection}. "
        f"Arquivos: {indexed_files_text}"
    )
    if st.session_state.suggested_questions:
        st.markdown("**Perguntas sugeridas para comecar**")
        for idx, suggested in enumerate(st.session_state.suggested_questions, start=1):
            st.write(f"{idx}. {suggested}")

if st.button("Processar documentos", type="primary"):
    if not uploaded_files:
        st.error("Envie pelo menos um arquivo antes de processar.")
    elif not _validate_runtime(
        provider=provider,
        api_key=active_api_key,
        chat_model=selected_chat_model,
        embedding_model=selected_embedding_model,
    ):
        st.stop()
    elif st.session_state.available_models_provider != provider:
        st.error("Carregue os modelos da API key para o provedor selecionado antes de processar.")
        st.stop()
    elif provider == "openai" and "embedding" not in selected_embedding_model:
        st.error("Para OpenAI, selecione um modelo de embeddings (ex.: text-embedding-3-small).")
        st.stop()
    elif provider == "google" and "embed" not in selected_embedding_model:
        st.error("Para Google, selecione um modelo de embeddings (ex.: models/text-embedding-004).")
        st.stop()
    else:
        try:
            all_chunks: List[dict] = []
            indexed_files: List[str] = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_uploaded_file(uploaded_file)
                chunks = chunk_text(
                    text=text,
                    chunk_size=650,
                    chunk_overlap=120,
                    metadata={"source": uploaded_file.name},
                )
                all_chunks.extend(chunks)
                indexed_files.append(uploaded_file.name)

            if not all_chunks:
                st.error("Nao foi possivel extrair conteudo util dos arquivos enviados.")
            else:
                texts = [c["content"] for c in all_chunks]
                embeddings = get_embeddings(
                    texts=texts,
                    provider=provider,
                    api_keys=api_keys,
                    embedding_model=selected_embedding_model,
                )

                collection_name = f"rag_chunks_{provider}"
                store = ChromaVectorStore(collection_name=collection_name, persist_dir=".chroma")
                store.upsert_chunks(chunks=all_chunks, embeddings=embeddings)

                st.session_state.index_ready = True
                st.session_state.indexed_docs = indexed_files
                st.session_state.indexed_provider = provider
                st.session_state.indexed_collection = collection_name
                st.session_state.suggested_questions = _suggest_questions_from_chunks(all_chunks, limit=3)
                st.success(f"Indexacao concluida com {len(all_chunks)} chunks.")
                st.markdown("**Sugestoes de perguntas com base no documento**")
                for idx, suggested in enumerate(st.session_state.suggested_questions, start=1):
                    st.write(f"{idx}. {suggested}")
        except Exception as exc:
            st.session_state.index_ready = False
            st.error(f"Falha ao processar documentos: {exc}")

question = st.text_input("Pergunta", placeholder="Pergunte algo sobre os documentos...")

if st.button("Perguntar"):
    if not st.session_state.index_ready:
        st.error("Processe os documentos antes de perguntar.")
    elif not question.strip():
        st.error("Digite uma pergunta valida.")
    elif st.session_state.indexed_provider and provider != st.session_state.indexed_provider:
        st.error(
            "O provedor atual difere do provedor usado na indexacao. "
            "Reprocesse os documentos com o mesmo provedor selecionado para garantir consistencia."
        )
    elif not _validate_runtime(
        provider=provider,
        api_key=active_api_key,
        chat_model=selected_chat_model,
        embedding_model=selected_embedding_model,
    ):
        st.stop()
    else:
        try:
            query_embedding = embed_query(
                question,
                provider=provider,
                api_keys=api_keys,
                embedding_model=selected_embedding_model,
            )

            store = ChromaVectorStore(
                collection_name=st.session_state.indexed_collection,
                persist_dir=".chroma",
            )
            diagnostic_results = store.search_similar(
                query_embedding=query_embedding,
                top_k=max(top_k, 8),
                similarity_threshold=0.0,
            )
            results = [row for row in diagnostic_results if float(row.get("similarity", 0.0)) >= similarity_threshold][:top_k]

            low_confidence_results = [
                row for row in diagnostic_results if float(row.get("similarity", 0.0)) >= 0.2
            ][:top_k]
            used_low_confidence_fallback = False

            # Evita falso-negativo quando o threshold configurado estiver muito alto.
            if not results and low_confidence_results:
                results = low_confidence_results
                used_low_confidence_fallback = True

            if not results:
                st.warning("Nao encontrei essa informacao no documento.")
            else:
                if used_low_confidence_fallback:
                    st.warning(
                        "Nao havia resultados acima do threshold configurado. "
                        "Usei os melhores trechos com similaridade moderada (>= 0.2)."
                    )

                context = "\n\n".join([r["content"] for r in results])
                answer = generate_answer(
                    question=question,
                    context=context,
                    provider=provider,
                    api_keys=api_keys,
                    chat_model=selected_chat_model,
                )

                st.subheader("Resposta")
                st.write(answer)

                st.subheader("Trechos fonte")
                for idx, row in enumerate(results, start=1):
                    source = row.get("metadata", {}).get("source", "desconhecido")
                    st.markdown(f"{idx}. Fonte: {source}")
                    st.code(row["content"][:1000])

            if diagnostic_mode:
                st.subheader("Diagnostico de recuperacao")
                if not diagnostic_results:
                    st.info("Nenhum chunk retornado pelo retriever.")
                else:
                    st.caption(
                        "A tabela abaixo mostra os chunks retornados pelo retriever e quais foram descartados pelo threshold atual."
                    )
                    for idx, row in enumerate(diagnostic_results, start=1):
                        similarity = float(row.get("similarity", 0.0))
                        source = row.get("metadata", {}).get("source", "desconhecido")
                        status = "APROVADO" if similarity >= similarity_threshold else "DESCARTADO"
                        st.markdown(
                            f"{idx}. [{status}] Fonte: {source} | Similaridade: {round(similarity, 4)}"
                        )
                        st.code(row.get("content", "")[:500])
        except Exception as exc:
            st.error(f"Falha ao gerar resposta: {exc}")
