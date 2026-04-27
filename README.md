# Chatbot RAG com PostgreSQL

MVP de chatbot de perguntas e respostas baseado em documentos, usando Retrieval Augmented Generation (RAG).

## Funcionalidades

- Upload de arquivos: PDF, TXT, DOCX, XLSX e CSV
- Extracao e chunking de texto
- Embeddings com OpenAI, Google ou Hugging Face
- Armazenamento vetorial no PostgreSQL com pgvector
- Busca Top-K por similaridade
- Resposta baseada apenas no contexto recuperado
- Exibicao de trecho fonte na interface

## Estrutura

- app.py
- ingestion/loader.py
- retriever/postgres_store.py
- llm/chain.py
- tests/test_loader.py
- requirements.txt

## Requisitos

- Python 3.12+
- PostgreSQL 14+
- Extensao pgvector habilitada

## Instalacao

1. Criar e ativar ambiente virtual
2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Variaveis e chaves

A interface permite informar chaves de API durante a execucao:

- OpenAI API Key
- Google API Key
- Hugging Face API Token

Tambem e necessario informar PostgreSQL DSN:

- Exemplo: postgresql://user:password@localhost:5432/ragdb

## Execucao

```bash
streamlit run app.py
```

## Testes

```bash
pytest
```

## Prompt de seguranca

O chatbot usa instrucao para responder somente com base no contexto.
Quando nao encontra resposta no contexto, retorna mensagem de fallback.

## Limitacoes do MVP

- Sem autenticacao
- Sem historico persistente de conversa
- Sem suporte a ranking hibrido
- Dependente de APIs externas para embeddings e geracao
