# PDF RAG Chatbot (Ollama)

A fast and efficient RAG (Retrieval-Augmented Generation) chatbot that answers questions from PDF documents using Ollama.

## Features

- 📄 Query PDF documents using natural language
- ⚡ Fast embeddings with FAISS vector store
- 🔄 Cached embeddings for instant loading on subsequent runs
- 🧠 Powered by Ollama's phi3 and bge-m3 models

## Setup

### Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the required models:
   ```bash
   ollama pull phi3
   ollama pull bge-m3
   ```

### Installation

1. Install Python dependencies:
   ```bash
   pip install streamlit langchain-community langchain-text-splitters pypdf faiss-cpu
   ```

## Usage

### Adding Documents to Knowledge Base

1. Place your PDF documents in the `knowledge_base/` directory:
   ```
   knowledge_base/
   ├── document1.pdf
   ├── document2.pdf
   └── your_document.pdf
   ```

2. Documents must be in PDF format and placed directly in the `knowledge_base/` directory

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run rag.py
   ```

2. Select a PDF from the dropdown menu

3. Ask questions about the selected document

4. The first time a PDF is processed, embeddings will be created and saved (this takes a few moments)

5. Subsequent queries on the same PDF will be much faster as embeddings are cached

## How It Works

1. PDFs are loaded from the `knowledge_base/` directory
2. Documents are split into chunks for better retrieval
3. Embeddings are created using the bge-m3 model
4. FAISS vector store enables fast similarity search
5. The phi3 model generates answers based on retrieved context

## Models

- **Answer Model**: phi3 - Optimized for low RAM usage and fast responses
- **Embedding Model**: bge-m3 - High-quality multilingual embeddings

## Notes

- Vector stores are saved with the prefix `vs_` followed by the PDF filename
- Cached embeddings are stored locally for faster subsequent queries
- Only content from the selected PDF is used to answer questions
