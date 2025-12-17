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
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Additional imports for Excel/CSV processing
import pandas as pd

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
    
    def _process_excel_file(self, file_path: str):
        """Process Excel file and convert to document format"""
        from langchain_core.documents import Document
        
        try:
            # Read Excel file with pandas
            df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            
            docs = []
            
            # Process each sheet
            for sheet_name, data in df.items():
                # Convert DataFrame to text format
                sheet_content = f"Sheet: {sheet_name}\n\n"
                
                # Add column headers
                headers = list(data.columns)
                sheet_content += "Columns: " + ", ".join(str(col) for col in headers) + "\n\n"
                
                # Add data rows
                for index, row in data.iterrows():
                    row_text = []
                    for col in headers:
                        value = row[col]
                        if pd.notna(value):  # Only add non-null values
                            row_text.append(f"{col}: {value}")
                    
                    if row_text:  # Only add rows with data
                        sheet_content += "Row " + str(index + 1) + ": " + " | ".join(row_text) + "\n"
                
                # Create document for this sheet
                doc = Document(
                    page_content=sheet_content,
                    metadata={
                        "source": file_path,
                        "sheet": sheet_name,
                        "type": "excel",
                        "rows": len(data),
                        "columns": len(headers)
                    }
                )
                docs.append(doc)
                
            logger.info(f"Processed Excel file {file_path} with {len(docs)} sheets")
            return docs
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            # Fallback to UnstructuredExcelLoader if available
            try:
                loader = UnstructuredExcelLoader(file_path)
                return loader.load()
            except Exception as e2:
                logger.error(f"Fallback Excel loader also failed: {e2}")
                raise e
    
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
            elif doc_type == "excel":
                # Enhanced Excel processing
                docs = self._process_excel_file(file_path)
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            if doc_type == "excel":
                # docs already processed for Excel
                pass
            else:
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
        supported_extensions = ['*.pdf', '*.txt', '*.csv', '*.xlsx', '*.xls']
        
        for extension in supported_extensions:
            pattern = os.path.join(DOCUMENTS_DIR, extension)
            files = glob.glob(pattern)
            
            for file_path in files:
                doc_name = os.path.basename(file_path)
                # Skip vector store folders and other non-document files
                if doc_name.startswith('vs_') or doc_name.startswith('combined_vs'):
                    continue
                    
                db_path = os.path.join(DOCUMENTS_DIR, f"vs_{doc_name}")
                
                # Check if document is already processed
                if doc_name not in self.documents and not os.path.exists(db_path):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    doc_type = {'.pdf': 'pdf', '.txt': 'txt', '.csv': 'csv', '.xlsx': 'excel', '.xls': 'excel'}.get(file_ext, 'txt')
                    
                    logger.info(f"Auto-loading document: {doc_name}")
                    self.add_document(file_path, doc_type)
                elif doc_name not in self.documents and os.path.exists(db_path):
                    # Document was processed before, just add to registry
                    file_ext = os.path.splitext(file_path)[1].lower()
                    doc_type = {'.pdf': 'pdf', '.txt': 'txt', '.csv': 'csv', '.xlsx': 'excel', '.xls': 'excel'}.get(file_ext, 'txt')
                    
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
    
    def search_knowledge_base(self, question: str, k: int = 3, min_similarity: float = 0.5) -> List[Dict]:
        """Search the knowledge base for relevant information with relaxed similarity threshold"""
        if not self.combined_vectordb:
            return []
        
        try:
            # Use similarity_search_with_score to get similarity scores
            docs_with_scores = self.combined_vectordb.similarity_search_with_score(question, k=k)
            results = []
            
            for doc, score in docs_with_scores:
                # FAISS returns distance (lower is better)
                # For embedding vectors, distance can be quite large (100s-1000s)
                # Convert to similarity score (0-1 range, higher is better)
                if score < 100:
                    similarity = max(0, 1.0 - (score / 100))  # Very good match
                elif score < 500:
                    similarity = max(0, 0.8 - (score / 1000))  # Good match  
                else:
                    similarity = max(0, 0.5 - (score / 2000))  # Moderate match
                
                # Check for keyword matches to boost relevance
                content_lower = doc.page_content.lower()
                question_words = question.lower().split()
                keyword_matches = sum(1 for word in question_words if len(word) > 3 and word in content_lower)
                
                # Boost similarity if keywords found
                if keyword_matches > 0:
                    keyword_boost = min(0.3, keyword_matches * 0.1)
                    similarity = min(1.0, similarity + keyword_boost)
                    logger.info(f"Keyword boost: +{keyword_boost:.3f} for {keyword_matches} matches")
                
                # Include results with relaxed threshold
                if similarity >= min_similarity or keyword_matches > 0:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': similarity
                    })
                    logger.info(f"Document included - Similarity: {similarity:.3f}, Distance: {score:.1f}, Keywords: {keyword_matches}")
                else:
                    logger.info(f"Document rejected - Similarity: {similarity:.3f}, Distance: {score:.1f}")
            
            # Always include at least one result if documents exist
            if not results and docs_with_scores:
                logger.info("No good matches found, including best available document")
                doc, score = docs_with_scores[0]
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity': 0.1  # Low but present
                })
                logger.info(f"Fallback document included - Distance: {score:.1f}")
            
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

class RAGSystem:
    """Standalone RAG System - No Web Interface"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.kb_manager = KnowledgeBaseManager()
        self.llm = OllamaLLM(model=ANSWER_MODEL)
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

    def ask_question(self, question: str, use_cache: bool = True, k: int = 3, force_fresh: bool = False) -> Dict:
        """Ask a question - Same question uses cache, New question checks all documents + APIs fresh"""
        start_time = time.time()
        
        # Check if this is a repeated question (exists in cache)
        cached = self.kb_manager.get_cached_answer(question)
        
        if cached and use_cache and not force_fresh:
            # REPEATED QUESTION - Use cache
            execution_time = time.time() - start_time
            self.kb_manager.log_operation("cached_answer", f"Question: {question[:50]}...", execution_time)
            print(f"🔄 REPEATED QUESTION - Retrieved from cache in {execution_time:.2f}s")
            
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
        
        # NEW QUESTION - Process fresh through all documents and APIs
        print(f"🆕 NEW QUESTION - Processing fresh through ALL documents (CSV, PDF, DOC) and APIs")
        print(f"🚫 No hallucination - Only knowledge base content allowed")
        
        # Check if this is obviously a math/calculation question that should skip documents
        math_patterns = [
            r'^\d+\s*[\+\-\*/]\s*\d+',  # Simple arithmetic: 5+7, 10*3
            r'^\d+$',  # Just a single number: 9, 46, 64 
            r'\bfibonacci\s+\d+',   # fibonacci 5
            r'\bfib\s+\d+',         # fib 10  
            r'\bfactors?\s+of\s+\d+',  # factors of 9
            r'what\s+is\s+factors?\s+of\s+\d+',  # what is factor of 9
            r'find\s+factors?\s+of\s+\d+',       # find factors of 64
            r'factorize\s+\d+',                  # factorize 25
        ]
        
        # Check for math keywords
        math_keywords = ['fibonacci', 'fib', 'factor', 'factors', 'factorize', 'calculate', 'math']
        has_math_keyword = any(keyword in question.lower() for keyword in math_keywords)
        has_math_pattern = any(re.search(pattern, question.lower().strip()) for pattern in math_patterns)
        
        is_obvious_math = has_math_pattern or (has_math_keyword and re.search(r'\d+', question))
        
        if is_obvious_math:
            print(f"🔢 Detected math/calculation question - Skipping documents, checking APIs directly")
            print(f"   🎯 Reason: Math keyword '{has_math_keyword}' or pattern match '{has_math_pattern}'")
            search_results = []
        else:
            # STEP 1: Check ALL document types first for non-math questions  
            print(f"🔍 Step 1: Searching ALL document types (PDF, CSV, Excel, TXT) for: {question}")
            search_results = self.kb_manager.search_knowledge_base(question, k=k)
        
        if search_results:
            # Found in documents - generate answer from documents
            print(f"📄 Found {len(search_results)} relevant document chunks from ALL file types")
            for i, result in enumerate(search_results):
                source = result['metadata'].get('source', 'unknown')
                similarity = result.get('similarity', 0)
                content_preview = result['content'][:100].replace('\n', ' ')
                file_type = "CSV" if source.endswith('.csv') else "Excel" if source.endswith(('.xlsx', '.xls')) else "PDF" if source.endswith('.pdf') else "Document"
                print(f"   📋 Chunk {i+1}: {source} ({file_type}) (similarity: {similarity:.3f}) - {content_preview}...")
            
            context = "\n\n".join([result['content'] for result in search_results])
            source_files = list(set([result['metadata'].get('source', 'unknown') for result in search_results]))
            
            prompt = f"""
You are a STRICT Knowledge Base Question Answering System.

ZERO HALLUCINATION POLICY - STRICT RULES:

1. You MUST answer ONLY using the provided Knowledge Base content below.
2. You are FORBIDDEN from using your own knowledge, training data, or external information.
3. You are FORBIDDEN from guessing, assuming, inferring, or hallucinating any information.
4. For data value queries, provide the EXACT value from the content.
5. For CSV/Excel data, provide the precise field value as shown.
6. If information is NOT explicitly present in the Knowledge Base content, respond exactly:
   "Not Found in Knowledge Base"
7. NO explanations, suggestions, or additional information beyond what's in the content.

Knowledge Base Content (CSV, PDF, Excel, Documents):
{context}

User Question: {question}

STRICT ANSWER (exact content only, no hallucination):
"""
            
            try:
                answer = self.llm.invoke(prompt)
                
                # ABSOLUTE STRICT validation - enforce exact compliance
                strict_not_found_phrases = [
                    "not found in knowledge base",
                    "information not available",
                    "not in the context",
                    "cannot find",
                    "no information",
                    "not provided",
                    "unable to answer",
                    "insufficient information",
                    "not mentioned",
                    "i don't know",
                    "i cannot",
                    "i'm not sure",
                    "unclear",
                    "uncertain",
                    "not sure",
                    "partially",
                    "might be",
                    "could be",
                    "seems like",
                    "appears to"
                ]
                
                answer_lower = answer.lower().strip()
                is_not_found = any(phrase in answer_lower for phrase in strict_not_found_phrases)
                # Allow short answers for data values (numbers, single words from CSV/Excel)
                is_too_short = len(answer.split()) < 2 and not any(char.isdigit() for char in answer)
                contains_uncertainty = any(word in answer_lower for word in ['maybe', 'possibly', 'likely', 'probably'])
                
                # STRICT VALIDATION: No hallucination allowed
                if is_not_found or is_too_short or contains_uncertainty:
                    print(f"📄 STRICT MODE: Answer rejected - '{answer[:50]}...'")
                    print("🔄 Reason: Not explicitly found in documents - checking APIs next")
                    # Continue to API check
                else:
                    # Found valid answer in documents - Cache it for repeated questions
                    self.kb_manager.cache_answer(question, answer, context, source_files)
                    
                    execution_time = time.time() - start_time
                    print(f"✅ NEW QUESTION: Answer found in documents in {execution_time:.2f}s")
                    print(f"📄 Document Sources: {', '.join(source_files)}")
                    print(f"💾 Cached for future repeated questions")
                    
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
        
        # STEP 2: Check ALL APIs if not found in documents
        print(f"🌐 Step 2: NEW QUESTION - Checking ALL APIs for: {question}")
        api_result = self.check_all_apis(question)
        
        if api_result["found"]:
            # Found in API - Cache for repeated questions
            execution_time = time.time() - start_time
            print(f"✅ NEW QUESTION: Answer found in {api_result['source']} in {execution_time:.2f}s")
            print(f"💾 Cached for future repeated questions")
            
            # Cache the API result for repeated questions
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
            print("🌐 NEW QUESTION: No calculation found in any APIs")
        
        # STEP 3: STRICT - Not found anywhere  
        print("❌ NEW QUESTION: Not found in documents (CSV, PDF, Excel, TXT) or APIs")
        print("📋 Responding: 'Not Found in Knowledge Base' - No hallucination")
        execution_time = time.time() - start_time
        
        # Cache the "not found" result to avoid repeated searches of same question
        self.kb_manager.cache_answer(question, "Not Found in Knowledge Base", "Question outside Knowledge Base scope", [])
        print(f"💾 Cached 'Not Found' result for repeated questions")
        
        return {
            "answer": "Not Found in Knowledge Base",
            "sources": [],
            "cached": False,
            "execution_time": execution_time,
            "found_in": "nowhere",
            "strict_mode": True
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