#!/usr/bin/env python3
"""
Test script to verify factorization functionality
"""

import sys
sys.path.append('apis')
from factorization_api import process_factorization

def test_factorization():
    """Test the factorization functionality"""
    print("🧪 Testing Factorization Logic")
    print("=" * 40)
    
    test_cases = [
        "9",
        "what is factor of 9", 
        "factors of 46",
        "find factors of 64",
        "25",
        "factorize 36"
    ]
    
    for test in test_cases:
        result = process_factorization(test)
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{status}: '{test}' -> {result['answer']}")
    
    print("=" * 40)
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_factorization()