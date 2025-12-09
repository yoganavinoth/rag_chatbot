#!/usr/bin/env python3
"""
Standalone RAG (Retrieval-Augmented Generation) Knowledge Base System
No Streamlit interface - Direct code-based interaction with knowledge base folder
"""

import os
import json
import hashlib
import time
import glob
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3
import logging
from pathlib import Path

# Core RAG imports
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 🔥 Configuration
ANSWER_MODEL = "phi3"
EMBED_MODEL = "bge-m3"
DOCUMENTS_DIR = "documents"  # Changed from knowledge_base to documents
CACHE_DB = "rag_cache.db"
LOG_FILE = "rag_debug.log"

# Multiple API endpoints configuration
API_ENDPOINTS = {
    "math_api": "http://localhost:8002",     # Math API (Fibonacci, complex math)
    "factorization_api": "http://localhost:8003",  # Factorization API (factors of 9, 46, 64)
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """Manages multiple documents in the knowledge base"""
    
    def __init__(self):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.documents = {}
        self.combined_vectordb = None
        self.init_cache_db()
        self.load_existing_documents()
        
    def init_cache_db(self):
        """Initialize SQLite database for caching answers"""
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cached_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT UNIQUE,
                question TEXT,
                answer TEXT,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_files TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS debug_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                operation TEXT,
                details TEXT,
                execution_time REAL
            )
        ''')
        conn.commit()
        conn.close()
        
    def get_question_hash(self, question: str) -> str:
        """Generate hash for question to use as cache key"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def cache_answer(self, question: str, answer: str, context: str, source_files: List[str]):
        """Cache answer for future use"""
        question_hash = self.get_question_hash(question)
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO cached_answers 
                (question_hash, question, answer, context, source_files)
                VALUES (?, ?, ?, ?, ?)
            ''', (question_hash, question, answer, context, json.dumps(source_files)))
            conn.commit()
            logger.info(f"Cached answer for question: {question[:50]}...")
        except Exception as e:
            logger.error(f"Error caching answer: {e}")
        finally:
            conn.close()
    
    def get_cached_answer(self, question: str) -> Optional[Dict]:
        """Retrieve cached answer if available"""
        question_hash = self.get_question_hash(question)
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT question, answer, context, source_files, timestamp
                FROM cached_answers WHERE question_hash = ?
            ''', (question_hash,))
            result = cursor.fetchone()
            if result:
                return {
                    'question': result[0],
                    'answer': result[1],
                    'context': result[2],
                    'source_files': json.loads(result[3]),
                    'timestamp': result[4]
                }
        except Exception as e:
            logger.error(f"Error retrieving cached answer: {e}")
        finally:
            conn.close()
        return None
    
    def log_operation(self, operation: str, details: str, execution_time: float):
        """Log operations for debugging"""
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO debug_logs (operation, details, execution_time)
                VALUES (?, ?, ?)
            ''', (operation, details, execution_time))
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging operation: {e}")
        finally:
            conn.close()
    
    def add_document(self, file_path: str, doc_type: str = "pdf") -> bool:
        """Add a document to the knowledge base"""
        start_time = time.time()
        try:
            if doc_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif doc_type == "txt":
                loader = TextLoader(file_path)
            elif doc_type == "csv":
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Create vector store for this document
            doc_name = os.path.basename(file_path)
            vectordb = FAISS.from_documents(chunks, self.embeddings)
            
            # Save individual document vector store
            db_path = os.path.join(DOCUMENTS_DIR, f"vs_{doc_name}")
            vectordb.save_local(db_path)
            
            self.documents[doc_name] = {
                'path': file_path,
                'type': doc_type,
                'db_path': db_path,
                'chunks': len(chunks),
                'added_at': datetime.now().isoformat()
            }
            
            # Rebuild combined vector store
            self._rebuild_combined_vectordb()
            
            execution_time = time.time() - start_time
            self.log_operation("add_document", f"Added {doc_name} ({doc_type})", execution_time)
            logger.info(f"Successfully added document: {doc_name}")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_operation("add_document_error", str(e), execution_time)
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def _rebuild_combined_vectordb(self):
        """Rebuild combined vector database with all documents"""
        try:
            all_vectordbs = []
            for doc_name, doc_info in self.documents.items():
                if os.path.exists(doc_info['db_path']):
                    vectordb = FAISS.load_local(
                        doc_info['db_path'],
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    all_vectordbs.append(vectordb)
            
            if all_vectordbs:
                self.combined_vectordb = all_vectordbs[0]
                for vectordb in all_vectordbs[1:]:
                    self.combined_vectordb.merge_from(vectordb)
                
                # Save combined vector store
                combined_path = os.path.join(DOCUMENTS_DIR, "combined_vs")
                self.combined_vectordb.save_local(combined_path)
                logger.info("Successfully rebuilt combined vector database")
            
        except Exception as e:
            logger.error(f"Error rebuilding combined vector database: {e}")
    
    def load_existing_documents(self):
        """Automatically scan and load documents from knowledge base folder"""
        supported_extensions = ['*.pdf', '*.txt', '*.csv']
        
        for extension in supported_extensions:
            pattern = os.path.join(DOCUMENTS_DIR, extension)
            files = glob.glob(pattern)
            
            for file_path in files:
                doc_name = os.path.basename(file_path)
                db_path = os.path.join(DOCUMENTS_DIR, f"vs_{doc_name}")
                
                # Check if document is already processed
                if doc_name not in self.documents and not os.path.exists(db_path):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    doc_type = {'.pdf': 'pdf', '.txt': 'txt', '.csv': 'csv'}.get(file_ext, 'txt')
                    
                    logger.info(f"Auto-loading document: {doc_name}")
                    self.add_document(file_path, doc_type)
                elif doc_name not in self.documents and os.path.exists(db_path):
                    # Document was processed before, just add to registry
                    file_ext = os.path.splitext(file_path)[1].lower()
                    doc_type = {'.pdf': 'pdf', '.txt': 'txt', '.csv': 'csv'}.get(file_ext, 'txt')
                    
                    # Count chunks by loading the vector store
                    try:
                        vectordb = FAISS.load_local(
                            db_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        chunk_count = vectordb.index.ntotal if hasattr(vectordb.index, 'ntotal') else 0
                    except:
                        chunk_count = 0
                    
                    self.documents[doc_name] = {
                        'path': file_path,
                        'type': doc_type,
                        'db_path': db_path,
                        'chunks': chunk_count,
                        'added_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        'auto_loaded': True
                    }
                    logger.info(f"Registered existing document: {doc_name}")
        
        # Rebuild combined vector store if we have documents
        if self.documents:
            self._rebuild_combined_vectordb()
            logger.info(f"Auto-loaded {len(self.documents)} documents from knowledge base folder")
    
    def scan_for_new_documents(self):
        """Scan for new documents added to the folder"""
        old_count = len(self.documents)
        self.load_existing_documents()
        new_count = len(self.documents)
        return new_count - old_count
    
    def search_knowledge_base(self, question: str, k: int = 3, min_similarity: float = 0.7) -> List[Dict]:
        """Search the knowledge base for relevant information with similarity threshold"""
        if not self.combined_vectordb:
            return []
        
        try:
            # Use similarity_search_with_score to get similarity scores
            docs_with_scores = self.combined_vectordb.similarity_search_with_score(question, k=k)
            results = []
            
            for doc, score in docs_with_scores:
                # FAISS returns distance (lower is better), so convert to similarity
                # Typical FAISS distance range is 0-2, where 0 = perfect match
                similarity = max(0, 1 - (score / 2.0))  # Convert distance to similarity (0-1)
                
                # Only include results above similarity threshold
                if similarity >= min_similarity:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': similarity
                    })
                    logger.info(f"Document match found - Similarity: {similarity:.3f}")
                else:
                    logger.info(f"Document rejected - Low similarity: {similarity:.3f}")
            
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

class RAGSystem:
    """Standalone RAG System - No Web Interface"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.kb_manager = KnowledgeBaseManager()
        self.llm = Ollama(model=ANSWER_MODEL)
        print(f"🤖 RAG System initialized with {len(self.kb_manager.documents)} documents")
    
    def check_factorization_api(self, question: str) -> Dict[str, Any]:
        """Check factorization API for factor-finding operations"""
        try:
            response = requests.post(
                f"{API_ENDPOINTS['factorization_api']}/factors",
                json={"question": question},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    return {
                        "found": True,
                        "answer": data["answer"],
                        "source": "Factorization API",
                        "api_type": "factorization"
                    }
        except requests.exceptions.RequestException as e:
            logger.warning(f"Factorization API not available: {e}")
        except Exception as e:
            logger.error(f"Error calling factorization API: {e}")
        
        return {"found": False}

    def check_math_api(self, question: str) -> Dict[str, Any]:
        """Check math API for complex mathematical calculations"""
        try:
            response = requests.post(
                f"{API_ENDPOINTS['math_api']}/calculate",
                json={"question": question},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    return {
                        "found": True,
                        "answer": data["answer"],
                        "source": "Math API",
                        "calculation_type": data.get("calculation_type", "unknown")
                    }
        except requests.exceptions.RequestException as e:
            logger.warning(f"Math API not available: {e}")
        except Exception as e:
            logger.error(f"Error calling math API: {e}")
        
        return {"found": False}
    


    def check_all_apis(self, question: str) -> Dict[str, Any]:
        """Check all available APIs in priority order"""
        # Priority order: Factorization API first (for factor operations), then Math API
        apis_to_check = [
            ("Factorization API", self.check_factorization_api),
            ("Math API", self.check_math_api),
        ]
        
        for api_name, api_function in apis_to_check:
            logger.info(f"Checking {api_name}...")
            result = api_function(question)
            if result["found"]:
                logger.info(f"Answer found in {api_name}")
                return result
            else:
                logger.info(f"Not found in {api_name}")
        
        return {"found": False}

    def ask_question(self, question: str, use_cache: bool = True, k: int = 3) -> Dict:
        """Ask a question with your requirement: Documents First -> APIs -> Not Found"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached = self.kb_manager.get_cached_answer(question)
            if cached:
                execution_time = time.time() - start_time
                self.kb_manager.log_operation("cached_answer", f"Question: {question[:50]}...", execution_time)
                print(f"💾 Retrieved cached answer in {execution_time:.2f}s")
                
                # Determine source type for cached answers
                source_files = cached['source_files']
                if any('API' in source for source in source_files):
                    found_in = 'api'
                else:
                    found_in = 'documents'
                
                return {
                    "answer": cached['answer'],
                    "sources": source_files,
                    "cached": True,
                    "timestamp": cached['timestamp'],
                    "execution_time": execution_time,
                    "found_in": found_in
                }
        
        # Check if this is obviously a math/calculation question that should skip documents
        math_patterns = [
            r'^\d+[\+\-\*/]\d+',  # 5+7, 10*3, etc.
            r'^\d+$',  # Just a number like 9, 46, 64
            r'fibonacci\s+\d+',   # fibonacci 5
            r'fib\s+\d+',         # fib 10
            r'factors?\s+of\s+\d+',  # factors of 9
            r'what\s+is\s+factors?\s+of\s+\d+',  # what is factor of 9
        ]
        
        is_obvious_math = any(re.search(pattern, question.lower().strip()) for pattern in math_patterns)
        
        if is_obvious_math:
            print(f"🔢 Detected math question - Skipping documents, checking APIs directly")
            search_results = []
        else:
            # STEP 1: Check documents first for non-math questions
            print(f"🔍 Step 1: Searching documents for: {question}")
            search_results = self.kb_manager.search_knowledge_base(question, k=k)
        
        if search_results:
            # Found in documents - generate answer from documents
            context = "\n\n".join([result['content'] for result in search_results])
            source_files = list(set([result['metadata'].get('source', 'unknown') for result in search_results]))
            
            prompt = f"""
Answer the question using only the provided context from the knowledge base.
If the answer is not in the context, say "Information not available in knowledge base".

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
            
            try:
                answer = self.llm.invoke(prompt)
                
                # Check if the answer indicates information is not available
                if "information not available" in answer.lower() or "not in the context" in answer.lower():
                    print("📄 Document search: Information not found in context")
                    # Continue to API check
                else:
                    # Found good answer in documents
                    if use_cache:
                        self.kb_manager.cache_answer(question, answer, context, source_files)
                    
                    execution_time = time.time() - start_time
                    print(f"✅ Answer found in documents in {execution_time:.2f}s")
                    print(f"📄 Document Sources: {', '.join(source_files)}")
                    
                    return {
                        "answer": answer,
                        "sources": source_files,
                        "cached": False,
                        "execution_time": execution_time,
                        "found_in": "documents"
                    }
                    
            except Exception as e:
                logger.error(f"Error generating answer from documents: {e}")
        else:
            print("📄 Document search: No relevant documents found")
        
        # STEP 2: Check All APIs if not found in documents
        print(f"🌐 Step 2: Checking all APIs for: {question}")
        api_result = self.check_all_apis(question)
        
        if api_result["found"]:
            # Found in one of the APIs
            execution_time = time.time() - start_time
            print(f"✅ Answer found in {api_result['source']} in {execution_time:.2f}s")
            
            # Cache the API result
            if use_cache:
                self.kb_manager.cache_answer(question, api_result["answer"], api_result["source"], [api_result["source"]])
            
            return {
                "answer": api_result["answer"],
                "sources": [api_result["source"]],
                "cached": False,
                "execution_time": execution_time,
                "found_in": "api",
                "api_source": api_result["source"],
                "calculation_type": api_result.get("calculation_type", api_result.get("api_type", "unknown"))
            }
        else:
            print("🌐 All APIs: No calculation found")
        
        # STEP 3: Not found anywhere
        print("❌ Step 3: Information not found in documents or any APIs")
        execution_time = time.time() - start_time
        
        not_found_answer = "Information not available in knowledge base or any APIs"
        
        return {
            "answer": not_found_answer,
            "sources": [],
            "cached": False,
            "execution_time": execution_time,
            "found_in": "nowhere"
        }
    
    def add_document(self, file_path: str) -> bool:
        """Add a document to the knowledge base"""
        file_ext = os.path.splitext(file_path)[1].lower()
        doc_type = {'.pdf': 'pdf', '.txt': 'txt', '.csv': 'csv'}.get(file_ext, 'txt')
        return self.kb_manager.add_document(file_path, doc_type)
    
    def scan_for_new_documents(self) -> int:
        """Scan for new documents in the knowledge base folder"""
        return self.kb_manager.scan_for_new_documents()
    
    def get_status(self) -> Dict:
        """Get system status"""
        cache_size = self._get_cache_size()
        return {
            "total_documents": len(self.kb_manager.documents),
            "documents": list(self.kb_manager.documents.keys()),
            "cache_size": cache_size,
            "documents_dir": DOCUMENTS_DIR
        }
    
    def _get_cache_size(self) -> int:
        """Get number of cached answers"""
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM cached_answers")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def list_documents(self):
        """List all documents in the knowledge base"""
        if not self.kb_manager.documents:
            print("📁 No documents found in knowledge base")
            return
        
        print(f"📚 Knowledge Base Documents ({len(self.kb_manager.documents)}):")
        print("-" * 50)
        for doc_name, doc_info in self.kb_manager.documents.items():
            status = "🔄 Auto-loaded" if doc_info.get('auto_loaded', False) else "📤 Manually added"
            print(f"{status} | {doc_name} | {doc_info['chunks']} chunks | {doc_info['type'].upper()}")
    
    def fibonacci(self, n: int) -> Dict:
        """Calculate Fibonacci sequence"""
        if n <= 0:
            return {"error": "Number must be positive"}
        
        fib_sequence = [0, 1]
        for i in range(2, n):
            fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
        
        return {
            "n": n,
            "sequence": fib_sequence[:n],
            "nth_number": fib_sequence[n-1] if n <= len(fib_sequence) else None
        }





def interactive_mode(rag_system):
    """Interactive question-answer mode"""
    print("\n" + "=" * 60)
    print("💬 Interactive Mode - Type your questions!")
    print("Commands: 'quit', 'exit', 'status', 'docs', 'scan'")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'status':
                status = rag_system.get_status()
                print(f"📊 Documents: {status['total_documents']}, Cache: {status['cache_size']}")
                continue
            elif question.lower() == 'docs':
                rag_system.list_documents()
                continue
            elif question.lower() == 'scan':
                new_docs = rag_system.scan_for_new_documents()
                print(f"🔍 Found {new_docs} new documents")
                continue
            elif question == '':
                continue
            
            # Ask the question
            result = rag_system.ask_question(question)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                cached_indicator = "💾 (cached)" if result.get('cached', False) else "🆕 (new)"
                print(f"\n🤖 Answer: {result['answer']}")
                print(f"⏱️  Time: {result['execution_time']:.2f}s {cached_indicator}")
                
                # Show sources with more detail
                sources = result.get('sources', [])
                if sources:
                    found_in = result.get('found_in', 'unknown')
                    if found_in == 'documents':
                        print(f"📄 Document Sources: {', '.join(sources)}")
                    elif found_in == 'api':
                        api_source = result.get('api_source', 'Unknown API')
                        print(f"🌐 API Source: {api_source}")
                        if result.get('calculation_type'):
                            print(f"🧮 Calculation Type: {result['calculation_type']}")
                    else:
                        print(f"📄 Sources: {', '.join(sources)}")
                else:
                    print("📄 Sources: None (not found anywhere)")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    """Main entry point"""
    import sys
    
    # Create knowledge base folder if it doesn't exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    if len(sys.argv) > 1:
        # Command line usage
        command = sys.argv[1].lower()
        
        if command == "ask" and len(sys.argv) > 2:
            # Direct question from command line
            rag = RAGSystem()
            question = " ".join(sys.argv[2:])
            result = rag.ask_question(question)
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(result['answer'])
        elif command == "interactive":
            rag = RAGSystem()
            interactive_mode(rag)
        else:
            print("Usage:")
            print("  python rag_pdf.py interactive             # Interactive mode")
            print("  python rag_pdf.py ask \"your question\"     # Ask single question")
    else:
        # Default: run interactive mode
        rag = RAGSystem()
        interactive_mode(rag)