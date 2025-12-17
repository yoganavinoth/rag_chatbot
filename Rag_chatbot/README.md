# 🤖 RAG Chatbot System

Simple and clean RAG (Retrieval-Augmented Generation) chatbot with document search and API integration.

## 📁 Project Structure

```
Rag_chatbot/
├── 📄 rag_pdf.py              # Main RAG system engine
├── 🌐 streamlit_app.py        # Web interface
├── 🚀 run_streamlit.py        # Application launcher
├── 📂 apis/                   # API servers
│   ├── math_api.py            # Math operations (fibonacci, calculations)
│   └── factorization_api.py   # Number factorization
└── 📂 documents/              # Knowledge base files
    ├── health.pdf             # Health information
    ├── metrics.csv            # Performance metrics
    ├── health_conditions.csv  # Medical conditions data
    ├── product_catalog.csv    # Product information
    ├── sample_data.xlsx       # Multi-sheet Excel data
    └── sample-1-10.pdf        # Additional documents
```

## 🚀 How to Run

1. **Start the system:**
   ```bash
   python run_streamlit.py
   ```

2. **Open web interface:**
   - URL: http://localhost:8501

## 🎯 Features

- 📚 **Multi-format support:** PDF, CSV, Excel (XLSX/XLS), TXT
- 🔍 **Smart search:** Document-first, then API fallback
- 🧮 **Math APIs:** Fibonacci, factorization, calculations
- 💾 **Smart caching:** Fresh processing vs cached results
- 🌐 **Clean web UI:** Simple Streamlit interface

## 💡 Usage Examples

- **Document queries:** "What is diabetes?" → Searches health PDFs/CSVs
- **Data queries:** "What is depth_score_pct?" → Gets value from CSV
- **Math queries:** "fibonacci 5" → Uses math API
- **Factor queries:** "factors of 46" → Uses factorization API

## ⚙️ System Flow

```
User Question → Documents → APIs → "Not Found"
```

1. Search documents (PDF, CSV, Excel)
2. If not found, check math/factorization APIs
3. Return "Not Found in Knowledge Base" if nowhere

---

**Simple, focused, and efficient RAG system!** 🎉