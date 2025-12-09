#!/usr/bin/env python3
"""
Demo script showing different types of questions you can ask the RAG system
"""

def show_question_examples():
    """Show different types of questions you can ask"""
    print("🤖 RAG System - Question Examples")
    print("=" * 50)
    
    print("\n📄 DOCUMENT QUESTIONS:")
    print("=" * 30)
    document_questions = [
        "what is health?",
        "tell me about diabetes",
        "what are the symptoms?",
        "health tips",
        "medicine information",
        "doctor advice",
        "treatment options"
    ]
    for q in document_questions:
        print(f"   ❓ {q}")
    
    print("\n🔢 FACTORIZATION API QUESTIONS:")
    print("=" * 35)
    factor_questions = [
        "9",
        "factors of 9",
        "what is factor of 9", 
        "find factors of 46",
        "factors of 64",
        "factorize 25",
        "what are the factors of 36",
        "prime factors of 12"
    ]
    for q in factor_questions:
        print(f"   ❓ {q}")
    
    print("\n🧮 MATH API QUESTIONS:")
    print("=" * 25)
    math_questions = [
        "fibonacci 5",
        "what is 5th fibonacci number",
        "fib 10",
        "fibonacci sequence",
        "calculate fibonacci 7",
        "fibonacci of 8",
        "math calculation"
    ]
    for q in math_questions:
        print(f"   ❓ {q}")
    
    print("\n⚡ WORKFLOW:")
    print("=" * 15)
    print("   1️⃣ System searches ALL documents first")
    print("   2️⃣ If not found → checks Factorization API") 
    print("   3️⃣ If not found → checks Math API")
    print("   4️⃣ If not found → returns 'Information not available'")
    
    print("\n🚀 HOW TO ASK:")
    print("=" * 15)
    print("   • Start the system: python start.py")
    print("   • Type any question from above examples")
    print("   • System will show which step found the answer")
    print("   • Type 'quit' to exit")
    
    print("\n✅ EXAMPLES OF GOOD QUESTIONS:")
    print("=" * 35)
    examples = [
        ("Document search", "what is diabetes symptoms"),
        ("Factorization API", "what is factor of 9"),
        ("Math API", "fibonacci 5"),
        ("Not found", "what is the weather today")
    ]
    
    for category, question in examples:
        print(f"   📝 {question:<25} → {category}")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to test! Run: python start.py")

if __name__ == "__main__":
    show_question_examples()