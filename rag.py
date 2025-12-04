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

st.title("📄 PDF RAG Chatbot (Ollama) • Ultra Fast Version")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_path = uploaded_pdf.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    DB_PATH = f"vs_{os.path.splitext(pdf_path)[0]}"

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
