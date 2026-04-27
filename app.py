import os
from typing import Dict, List

import streamlit as st

from ingestion.loader import chunk_text, extract_text_from_uploaded_file
from llm.chain import embed_query, generate_answer, get_embeddings
from retriever.postgres_store import PostgresVectorStore


st.set_page_config(page_title="Chatbot RAG", page_icon="RAG", layout="wide")

st.title("Chatbot de Perguntas e Respostas (RAG)")
st.caption("Responda perguntas com base nos documentos enviados.")

with st.sidebar:
    st.header("Configuracao")
    provider = st.selectbox(
        "Provedor de IA",
        options=["openai", "google", "huggingface"],
        index=0,
    )

    st.subheader("Chaves de API")
    openai_key = st.text_input("OpenAI API Key", type="password")
    google_key = st.text_input("Google API Key", type="password")
    hf_key = st.text_input("Hugging Face API Token", type="password")

    postgres_dsn = st.text_input(
        "PostgreSQL DSN",
        value=os.getenv("POSTGRES_DSN", ""),
        help="Exemplo: postgresql://user:password@localhost:5432/ragdb",
    )

    top_k = st.slider("Top-K", min_value=3, max_value=5, value=4)
    similarity_threshold = st.slider("Threshold de similaridade", 0.0, 1.0, 0.6, 0.05)

st.divider()

uploaded_files = st.file_uploader(
    "Upload de documentos",
    type=["pdf", "txt", "docx", "xlsx", "csv"],
    accept_multiple_files=True,
)

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

api_keys: Dict[str, str] = {
    "openai": openai_key,
    "google": google_key,
    "huggingface": hf_key,
}

if st.button("Processar documentos", type="primary"):
    if not uploaded_files:
        st.error("Envie pelo menos um arquivo antes de processar.")
    elif not postgres_dsn:
        st.error("Informe o PostgreSQL DSN para armazenar embeddings.")
    elif not api_keys.get(provider):
        st.error(f"Informe a chave do provedor selecionado: {provider}.")
    else:
        try:
            all_chunks: List[dict] = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_uploaded_file(uploaded_file)
                chunks = chunk_text(
                    text=text,
                    chunk_size=650,
                    chunk_overlap=120,
                    metadata={"source": uploaded_file.name},
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("Nao foi possivel extrair conteudo util dos arquivos enviados.")
            else:
                texts = [c["content"] for c in all_chunks]
                embeddings = get_embeddings(texts=texts, provider=provider, api_keys=api_keys)

                store = PostgresVectorStore(dsn=postgres_dsn)
                store.ensure_schema()
                store.upsert_chunks(chunks=all_chunks, embeddings=embeddings)

                st.session_state.index_ready = True
                st.success(f"Indexacao concluida com {len(all_chunks)} chunks.")
        except Exception as exc:
            st.session_state.index_ready = False
            st.error(f"Falha ao processar documentos: {exc}")

question = st.text_input("Pergunta", placeholder="Pergunte algo sobre os documentos...")

if st.button("Perguntar"):
    if not st.session_state.index_ready:
        st.error("Processe os documentos antes de perguntar.")
    elif not question.strip():
        st.error("Digite uma pergunta valida.")
    elif not postgres_dsn:
        st.error("Informe o PostgreSQL DSN.")
    elif not api_keys.get(provider):
        st.error(f"Informe a chave do provedor selecionado: {provider}.")
    else:
        try:
            query_embedding = embed_query(
                question,
                provider=provider,
                api_keys=api_keys,
            )

            store = PostgresVectorStore(dsn=postgres_dsn)
            results = store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            if not results:
                st.warning("Nao encontrei essa informacao no documento.")
            else:
                context = "\n\n".join([r["content"] for r in results])
                answer = generate_answer(
                    question=question,
                    context=context,
                    provider=provider,
                    api_keys=api_keys,
                )

                st.subheader("Resposta")
                st.write(answer)

                st.subheader("Trechos fonte")
                for idx, row in enumerate(results, start=1):
                    source = row.get("metadata", {}).get("source", "desconhecido")
                    similarity = round(float(row.get("similarity", 0.0)), 4)
                    st.markdown(f"{idx}. Fonte: {source} | Similaridade: {similarity}")
                    st.code(row["content"][:1000])
        except Exception as exc:
            st.error(f"Falha ao gerar resposta: {exc}")
