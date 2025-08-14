# YouTube Video Q&A with RAG
### YouTube Transcript Question Answering with RAG and HuggingFace
A tool to perform question-answering on YouTube video transcripts using Retrieval-Augmented Generation (RAG) powered by HuggingFace models and LangChain.

---

## Table of Contents

- [Architecture](#architecture)
- [Usage](#usage)
- [Acknowledgment](#Acknowledgment)
---

## Architecture

This project combines multiple components to create an efficient YouTube video Q&A system:

1. **Transcript Fetching**  
   Uses `youtube_transcript_api` to fetch English transcripts of YouTube videos.

2. **Text Processing**  
   Splits the transcript into manageable chunks with LangChain's `RecursiveCharacterTextSplitter`.

3. **Vector Embeddings and Indexing**  
   Converts text chunks into dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them with FAISS for similarity search.

4. **Retriever**  
   A vector retriever that fetches the most relevant transcript chunks based on similarity to the user's question.

5. **Large Language Model (LLM)**  
   Uses a HuggingFace hosted endpoint (e.g., `HuggingFaceH4/zephyr-7b-beta`) via `ChatHuggingFace` to generate answers based on retrieved context.

6. **Prompting and Chain Execution**  
   Constructs a prompt combining context and user question, runs through a chain pipeline with LangChain's `Runnable` components, and parses the output as the final answer.

---

## Usage

Provide a YouTube video Link to fetch the transcript and ask questions about the content.

## Acknowledgment

- `youtube-transcript-api`  
- `LangChain`  
- `HuggingFace`  
- `FAISS`  
