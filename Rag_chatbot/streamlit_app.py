#!/usr/bin/env python3
"""
Streamlit RAG Chatbot with Enhanced Features
- PDF Table Extraction
- Excel Multi-Sheet & Multi-Column Search
- Min/Max/Range Column Detection
"""

import streamlit as st
import os
from rag_pdf import RAGSystem
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Enhanced",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("🚀 Initializing RAG System..."):
        st.session_state.rag_system = RAGSystem()
        st.session_state.chat_history = []

# Title and description
st.title("🤖 RAG Chatbot")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    rag = st.session_state.rag_system
    status = rag.get_status()
    
    st.metric("Total Documents", status['total_documents'])
    st.metric("Cached Answers", status['cache_size'])
    
    st.divider()
    
    st.header("⚙️ Settings")
    use_cache = st.checkbox("Use Cache", value=True, help="Use cached answers for faster responses")
    k_results = st.slider("Search Results (k)", min_value=3, max_value=10, value=5, 
                         help="Number of relevant chunks to retrieve")
    
    if st.button("🔄 Scan for New Documents"):
        with st.spinner("Scanning..."):
            new_docs = rag.scan_for_new_documents()
            if new_docs > 0:
                st.success(f"Found {new_docs} new documents!")
                st.rerun()
            else:
                st.info("No new documents found")

# Main chat interface
st.header("💬 Chat with Your Documents")

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat['question'])
    
    with st.chat_message("assistant"):
        st.write(chat['answer'])
        if chat.get('cached'):
            st.caption("💾 Cached response")
        if chat.get('source_files'):
            with st.expander("📄 Source Files"):
                for source in chat['source_files']:
                    st.write(f"- {os.path.basename(source)}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            start_time = time.time()
            result = st.session_state.rag_system.ask_question(
                prompt, 
                use_cache=use_cache,
                k=k_results
            )
            elapsed = time.time() - start_time
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.write(result['answer'])
            
            # Show metadata
            col1, col2 = st.columns([2, 1])
            with col1:
                if result.get('cached'):
                    st.caption("💾 Cached response")
                else:
                    st.caption(f"🆕 Generated in {elapsed:.2f}s")
            
            with col2:
                if result.get('source_files'):
                    with st.expander("📄 Sources"):
                        for source in result['source_files']:
                            st.write(f"- {os.path.basename(source)}")
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': prompt,
                'answer': result['answer'],
                'cached': result.get('cached', False),
                'source_files': result.get('source_files', [])
            })
