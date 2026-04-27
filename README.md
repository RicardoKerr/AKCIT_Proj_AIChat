# Chatbot RAG com ChromaDB Local

MVP de chatbot de perguntas e respostas baseado em documentos, usando Retrieval Augmented Generation (RAG).

## Funcionalidades

- Upload de arquivos: PDF, TXT, DOCX, XLSX e CSV
- Extracao e chunking de texto
- Embeddings com OpenAI, Google ou Hugging Face
- Armazenamento vetorial local com ChromaDB
- Busca Top-K por similaridade
- Resposta baseada apenas no contexto recuperado
- Exibicao de trecho fonte na interface

## Estrutura

- app.py
- ingestion/loader.py
- retriever/chroma_store.py
- llm/chain.py
- tests/test_loader.py
- requirements.txt

## Requisitos

- Python 3.12+

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

Nao e necessario configurar banco externo para avaliacao.
Os embeddings sao armazenados localmente na pasta `.chroma`.

## Execucao

```bash
streamlit run app.py
```

Fluxo de avaliacao simples:

1. Instalar dependencias
2. Rodar Streamlit
3. Enviar documento e testar perguntas

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
