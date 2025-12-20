#!/usr/bin/env python3
"""
Launcher script for Streamlit RAG Chatbot
Run this file: python run_streamlit.py
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit app"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app = os.path.join(script_dir, "streamlit_app.py")
    
    # Check if streamlit_app.py exists
    if not os.path.exists(streamlit_app):
        print("❌ Error: streamlit_app.py not found!")
        sys.exit(1)
    
    print("🚀 Starting Enhanced RAG Chatbot...")
    print("📊 Features: PDF Tables ✅ | Excel Multi-Sheet ✅ | Min/Max Detection ✅")
    print("-" * 70)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_app])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
