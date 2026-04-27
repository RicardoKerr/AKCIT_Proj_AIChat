# Chatbot RAG com ChromaDB Local

MVP de chatbot de perguntas e respostas baseado em documentos, usando Retrieval Augmented Generation (RAG).

## Descricao

O sistema permite enviar documentos, indexar o conteudo em uma base vetorial local e responder perguntas com base apenas no contexto recuperado desses documentos.

## Tecnologias

- Python 3.12+
- Streamlit
- ChromaDB (vetor local)
- OpenAI API
- Google Generative AI API
- Hugging Face Inference API
- PyPDF, pandas, python-docx

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

## Uso

1. Escolha o provedor de IA na barra lateral.
2. Informe a API key do provedor.
3. Carregue os modelos disponiveis e selecione um modelo de embeddings e um modelo de resposta.
4. Envie um ou mais documentos.
5. Clique em Processar documentos.
6. Digite uma pergunta e clique em Perguntar.

Fluxo de avaliacao simples:

1. Instalar dependencias
2. Rodar Streamlit
3. Enviar documento e testar perguntas

## Testes

```bash
pytest
```

## Secao de IA

- IA para embeddings: gera vetores dos chunks para busca semantica.
- IA para resposta: gera a resposta final a partir dos trechos recuperados.
- Controle anti-alucinacao: prompt restringe resposta ao contexto e retorna fallback quando nao encontra informacao.

## Prompt de seguranca

O chatbot usa instrucao para responder somente com base no contexto.
Quando nao encontra resposta no contexto, retorna mensagem de fallback.

## Limitacoes do MVP

- Sem autenticacao
- Sem historico persistente de conversa
- Sem suporte a ranking hibrido
- Dependente de APIs externas para embeddings e geracao

## Exemplos de perguntas

- O que o documento diz sobre MVP?
- Quais pontos principais o documento apresenta?
- Quais recomendacoes praticas aparecem no texto?
