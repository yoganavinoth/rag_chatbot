import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 🔥 Best models for your system (low RAM + fast)
ANSWER_MODEL = "phi3"
EMBED_MODEL = "bge-m3"
KNOWLEDGE_BASE_DIR = "knowledge_base"

st.title("📄 PDF RAG Chatbot (Ollama) • Ultra Fast Version")

# Get list of PDFs from knowledge base directory
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    st.error(f"❌ Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Please create it and add PDF files.")
    st.stop()

try:
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith('.pdf')]
except OSError as e:
    st.error(f"❌ Error accessing knowledge base directory: {e}")
    st.stop()

if not pdf_files:
    st.warning(f"⚠️ No PDF files found in '{KNOWLEDGE_BASE_DIR}' directory. Please add PDF files to the knowledge base.")
    st.info("💡 Place your PDF documents in the 'knowledge_base' directory and refresh the page.")
    st.stop()

# Select PDF from knowledge base
selected_pdf = st.selectbox("Select a PDF from knowledge base", pdf_files)

if selected_pdf:
    pdf_path = os.path.join(KNOWLEDGE_BASE_DIR, selected_pdf)
    pdf_name = os.path.splitext(selected_pdf)[0]
    DB_PATH = f"vs_{pdf_name}"

    st.write("🔄 Loading PDF...")
    docs = PyPDFLoader(pdf_path).load()

    if os.path.exists(DB_PATH):
        st.write("⚡ Loading saved embeddings (very fast)...")
        vectordb = FAISS.load_local(
            DB_PATH,
            OllamaEmbeddings(model=EMBED_MODEL),
            allow_dangerous_deserialization=True
        )
    else:
        st.write("⏳ Creating embeddings (first time only)...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL))
        vectordb.save_local(DB_PATH)
        st.success("✔ Embeddings created and saved")

    llm = Ollama(model=ANSWER_MODEL)

    question = st.text_input("Ask a question from the PDF")
    if question:
        docs = vectordb.similarity_search(question, k=1)  # fastest + accurate
        context = docs[0].page_content

        prompt = f"""
Answer only using the PDF content below.
If the answer is not present, reply exactly: "Not in PDF".

PDF CONTENT:
{context}

QUESTION: {question}

ANSWER:
"""
        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt)

        st.write("🤖 *AI Answer:*")
        st.success(answer)
