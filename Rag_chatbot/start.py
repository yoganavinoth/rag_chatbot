#!/usr/bin/env python3
"""
Single command to start everything - RAG system with all APIs
No need for separate terminals!
"""

import subprocess
import sys
import time
import os
from threading import Thread

def start_api_server(script_path, port_name):
    """Start an API server in background"""
    try:
        print(f"🚀 Starting {port_name}...")
        subprocess.run([sys.executable, script_path], cwd=os.getcwd())
    except Exception as e:
        print(f"❌ Error starting {port_name}: {e}")

def main():
    print("🎯 Starting RAG System with All APIs")
    print("=" * 50)
    
    # Start API servers in background threads
    print("🌐 Starting API servers...")
    
    # Start Factorization API
    factorization_thread = Thread(
        target=start_api_server, 
        args=("apis/factorization_api.py", "Factorization API (Port 8003)"),
        daemon=True
    )
    factorization_thread.start()
    
    # Start Math API
    math_thread = Thread(
        target=start_api_server,
        args=("apis/math_api.py", "Math API (Port 8002)"),
        daemon=True
    )
    math_thread.start()
    
    # Wait for APIs to start
    print("⏳ Waiting for APIs to initialize...")
    time.sleep(3)
    
    print("✅ APIs started! Now starting RAG system...")
    print("=" * 50)
    
    # Import and start RAG system
    try:
        from rag_pdf import RAGSystem, interactive_mode
        
        # Initialize RAG system
        rag = RAGSystem()
        
        print("\n💬 RAG System Ready!")
        print("📋 Flow: Documents First → APIs → Not Found")
        print("🌐 APIs: Factorization API (8003) + Math API (8002)")
        print("📄 Documents: documents/ folder")
        print("=" * 50)
        
        # Start interactive mode
        interactive_mode(rag)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()