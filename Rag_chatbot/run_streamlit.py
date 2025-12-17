#!/usr/bin/env python3
"""
Quick Streamlit Launcher - Direct Web Interface
"""

import subprocess
import sys
import os
import time

def check_apis():
    """Check if APIs are running"""
    import requests
    apis = {
        'Math API': 'http://localhost:8002',
        'Factorization API': 'http://localhost:8003'
    }
    
    running = []
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                running.append(name)
        except:
            pass
    
    return running

def start_apis():
    """Start API servers"""
    print("🚀 Starting API servers...")
    
    # Start both APIs
    factorization = subprocess.Popen([sys.executable, "apis/factorization_api.py"])
    math_api = subprocess.Popen([sys.executable, "apis/math_api.py"])
    
    print("⏳ Waiting 5 seconds for APIs to start...")
    time.sleep(5)
    
    return factorization, math_api

def main():
    print("🌐 RAG Chatbot - Streamlit Web Interface")
    print("="*45)
    
    # Check if APIs are already running
    running_apis = check_apis()
    
    if len(running_apis) < 2:
        print(f"📊 APIs running: {running_apis}")
        print("🔄 Starting missing APIs...")
        start_apis()
    else:
        print("✅ All APIs already running")
    
    # Start Streamlit
    print("\n🚀 Starting Streamlit Web Interface...")
    print("🌐 Opening at: http://localhost:8502")
    print("\n💡 Tip: Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8502",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()