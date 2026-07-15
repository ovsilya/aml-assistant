# AML Assistant — RAG Compliance Chatbot (Swiss AML & Sanctions)

A Retrieval-Augmented Generation (RAG) assistant that answers **Anti-Money-Laundering (AML) and sanctions** regulatory questions for the Swiss banking industry. It grounds a **GPT-4o tool-calling agent** in a **Pinecone**-indexed corpus of regulatory PDFs, so compliance professionals get fast, source-backed answers instead of manually searching dense regulation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Ingestion Pipeline](#ingestion-pipeline-vector_storepy)
- [Conversational Agent](#conversational-agent-apppy)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Setup & Usage](#setup--usage)
- [Tech Stack](#tech-stack)

## Overview

The system has two parts:
1. an **ingestion pipeline** (`vector_store.py`) that loads regulatory PDFs from Google Drive, chunks and embeds them, and indexes them in Pinecone; and
2. a **conversational agent** (`app.py`) — a command-line chat loop where a GPT-4o tool-calling agent retrieves relevant regulation on demand and answers in context, with session memory.

## Features

- **RAG architecture** for contextual, source-grounded answers.
- **Tool-calling agent** (LangChain) that decides when to query the knowledge base.
- **Pinecone** serverless vector search with a similarity **score threshold** to filter low-relevance context.
- **Source traceability** — file id/name metadata carried on every chunk (important for compliance/citation).
- **PDF ingestion from Google Drive** via a GCP service account.
- **Session-based chat history** for multi-turn context.
- **Compliance-focused prompt engineering.**

## Architecture

```
Google Drive (regulatory PDFs)
        │  (service account, recursive folder scan)
        ▼
  PyPDFLoader → RecursiveCharacterTextSplitter (chunk 500 / overlap 100)
        │  + metadata (file id, filename)
        ▼
  OpenAI text-embedding-3-small (1536-dim)  →  Pinecone index "aml-assistant"
                                               (serverless, cosine, AWS us-east-1)
        ▲
        │ retriever tool (similarity score threshold, k=5, threshold=0.3)
        │
  GPT-4o tool-calling agent (LangChain AgentExecutor)
        │  + RunnableWithMessageHistory (per-session ChatMessageHistory)
        ▼
   CLI chat loop (app.py)
```

## Ingestion Pipeline (`vector_store.py`)

1. Authenticate to **Google Drive** with a GCP service account; recursively find PDFs.
2. Download each PDF and parse with **`PyPDFLoader`**.
3. Chunk with **`RecursiveCharacterTextSplitter`** (size 500, overlap 100); attach `file id` + `filename` metadata.
4. Embed with **`text-embedding-3-small`** (1536-dim).
5. Create the Pinecone **serverless** index `aml-assistant` if it doesn't exist (dimension 1536, **cosine**, AWS `us-east-1`), wait until ready, and upsert.

## Conversational Agent (`app.py`)

- **Agent:** LangChain **tool-calling agent** (`create_tool_calling_agent` + `AgentExecutor`).
- **Retriever tool:** built on the Pinecone vector store — search type *similarity score threshold*, `k=5`, `score_threshold=0.3`.
- **Memory:** `RunnableWithMessageHistory` (`input_messages_key="input"`, `history_messages_key="chat_history"`); in-memory `ChatMessageHistory` per session (UUID session ids).
- **Prompts:** system prompt from `system_prompt_main.txt`; tool prompt from `search_tool_prompt.txt`.
- **Entrypoint:** a `chat()` CLI loop.

## Project Structure

- `app.py` — CLI conversational agent (GPT-4o + retriever tool + session memory).
- `vector_store.py` — Google Drive → Pinecone ingestion pipeline.
- `system_prompt_main.txt`, `system_prompt.txt` — system prompts.
- `search_tool_prompt.txt` — retriever tool prompt.

## Environment Variables

- **`OPENAI_API_KEY`** — OpenAI (GPT-4o + embeddings).
- **`PINECONE_API_KEY`** — Pinecone vector database.

(A GCP service-account credential file is also required for Google Drive access.)

## Setup & Usage

```bash
git clone https://github.com/ovsilya/aml-assistant.git
cd aml-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# set OPENAI_API_KEY, PINECONE_API_KEY, and provide the GCP service-account file

# 1) build / update the vector index from Google Drive PDFs
python vector_store.py

# 2) start the compliance chat assistant
python app.py
```

## Tech Stack

Python · LangChain (tool-calling agent, `RunnableWithMessageHistory`) · OpenAI **GPT-4o** + `text-embedding-3-small` · **Pinecone** (serverless vector DB) · `PyPDFLoader` · Google Drive API.
