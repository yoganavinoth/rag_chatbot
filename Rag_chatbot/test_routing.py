#!/usr/bin/env python3
"""
Test script to verify Math API routing is working correctly
"""

import sys
import os
sys.path.append('.')
from rag_pdf import RAGSystem

def test_math_routing():
    """Test that math questions go to APIs and not documents"""
    print("🧪 Testing Math API Routing")
    print("=" * 50)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Test cases that should go directly to APIs
    math_questions = [
        "5+7",
        "9",
        "fibonacci 5",
        "factors of 9",
        "what is factor of 46",
        "fib 10"
    ]
    
    for question in math_questions:
        print(f"\n🔍 Testing: '{question}'")
        print("-" * 30)
        
        try:
            result = rag.ask_question(question, use_cache=False)
            source = result.get('found_in', 'unknown')
            answer = result.get('answer', 'No answer')
            
            if 'API' in str(result.get('sources', [])):
                print(f"✅ CORRECT: Found in API")
                print(f"📄 Sources: {result['sources']}")
                print(f"🤖 Answer: {answer}")
            else:
                print(f"❌ ISSUE: Found in {source}")
                print(f"📄 Sources: {result['sources']}")
                print(f"🤖 Answer: {answer}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_math_routing()