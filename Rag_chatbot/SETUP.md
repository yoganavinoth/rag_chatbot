# RAG Chatbot Setup Guide

## 📋 **Requirements**
- Python 3.8+
- Ollama installed and running
- 8GB+ RAM recommended

## 🚀 **Installation Steps**

### **1. Clone/Download Project**
```bash
# If you have the project folder, navigate to it
cd d:\Yogana\projects\Rag_chatbot
```

### **2. Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

### **4. Install Ollama Models**
```bash
# Install required AI models
ollama pull phi3
ollama pull bge-m3
```

### **5. Setup Documents**
```bash
# Place your PDF documents in the documents/ folder
# Example: documents/health.pdf, documents/sample.pdf
```

### **6. Run the System**
```bash
# Start the complete RAG system
python start.py
```

## 🔧 **Alternative Installation (Individual Packages)**

If requirements.txt doesn't work, install packages individually:

```bash
# Core AI packages
pip install langchain langchain-community ollama

# Vector database
pip install faiss-cpu sentence-transformers

# Document processing  
pip install pypdf python-docx

# Web framework
pip install fastapi uvicorn streamlit

# Utilities
pip install requests numpy pandas pydantic python-dotenv colorama tqdm
```

## 📂 **Project Structure**
```
Rag_chatbot/
├── requirements.txt          # This file
├── start.py                 # Main startup script
├── rag_pdf.py              # RAG system core
├── documents/              # Put your PDFs here
│   ├── health.pdf
│   └── sample-1-10.pdf
└── apis/                   # API services
    ├── math_api.py         # Math calculations
    └── factorization_api.py # Factor calculations
```

## ✅ **Verification**
After installation, test with:
```bash
python test_factorization.py
```

## 🌐 **API Endpoints**
- **Math API**: http://localhost:8002/docs
- **Factorization API**: http://localhost:8003/docs

## 🤖 **Usage**
1. Run: `python start.py`
2. Ask questions like:
   - "what is diabetes" (documents)
   - "fibonacci 5" (math API)  
   - "factors of 9" (factorization API)
3. Type 'quit' to exit

## 🐛 **Troubleshooting**
- **Ollama not found**: Install Ollama from https://ollama.ai
- **Port conflicts**: Kill processes using ports 8002, 8003
- **Model errors**: Run `ollama pull phi3` and `ollama pull bge-m3`