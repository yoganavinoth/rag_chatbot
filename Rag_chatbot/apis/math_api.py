#!/usr/bin/env python3
"""
Math API for RAG Chatbot
Handles mathematical calculations like Fibonacci, arithmetic operations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uvicorn
import re
import math

app = FastAPI(
    title="🧮 Math Calculation API",
    description="Mathematical operations API for RAG chatbot integration",
    version="1.0.0"
)

# Request/Response models
class MathQuery(BaseModel):
    question: str

class MathResponse(BaseModel):
    question: str
    answer: str
    calculation_type: str
    result: Union[int, float, List[int], str]
    success: bool

# Math calculation functions
def fibonacci_sequence(n: int) -> List[int]:
    """Generate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def fibonacci_nth(n: int) -> int:
    """Get nth Fibonacci number"""
    if n <= 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b
    return b

def evaluate_arithmetic(expression: str) -> Union[int, float]:
    """Safely evaluate arithmetic expressions"""
    # Remove spaces and validate characters
    expression = expression.replace(" ", "")
    
    # Only allow numbers, operators, and parentheses
    if not re.match(r'^[0-9+\-*/().]+$', expression):
        raise ValueError("Invalid characters in expression")
    
    try:
        # Use eval safely for basic arithmetic
        result = eval(expression)
        return result
    except:
        raise ValueError("Invalid arithmetic expression")

def process_math_question(question: str) -> Dict[str, Any]:
    """Process mathematical questions and return appropriate response"""
    question_lower = question.lower().strip()
    
    # Fibonacci patterns
    fibonacci_patterns = [
        r'fibonacci\s*(\d+)',
        r'fib\s*(\d+)',
        r'fibonacci\s*sequence\s*(\d+)',
        r'(\d+)\s*fibonacci',
        r'fibonacci\s*of\s*(\d+)',
    ]
    
    for pattern in fibonacci_patterns:
        match = re.search(pattern, question_lower)
        if match:
            n = int(match.group(1))
            if n > 50:  # Limit for performance
                return {
                    "answer": f"Fibonacci calculation limited to 50 terms. Requested: {n}",
                    "calculation_type": "fibonacci_limit",
                    "result": "limit_exceeded",
                    "success": False
                }
            
            if "sequence" in question_lower:
                fib_seq = fibonacci_sequence(n)
                return {
                    "answer": f"Fibonacci sequence of {n} terms: {fib_seq}",
                    "calculation_type": "fibonacci_sequence",
                    "result": fib_seq,
                    "success": True
                }
            else:
                fib_num = fibonacci_nth(n)
                return {
                    "answer": f"The {n}th Fibonacci number is {fib_num}",
                    "calculation_type": "fibonacci_nth",
                    "result": fib_num,
                    "success": True
                }
    
    # Arithmetic patterns
    arithmetic_patterns = [
        r'(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)',
        r'calculate\s*(.+)',
        r'what\s*is\s*(.+)',
        r'solve\s*(.+)',
    ]
    
    for pattern in arithmetic_patterns:
        match = re.search(pattern, question_lower)
        if match:
            expression = match.group(1).strip()
            try:
                result = evaluate_arithmetic(expression)
                return {
                    "answer": f"{expression} = {result}",
                    "calculation_type": "arithmetic",
                    "result": result,
                    "success": True
                }
            except ValueError as e:
                return {
                    "answer": f"Cannot calculate '{expression}': {str(e)}",
                    "calculation_type": "arithmetic_error",
                    "result": "error",
                    "success": False
                }
    
    # Check for direct arithmetic expressions like "5+7", "12*3", etc.
    direct_arithmetic = re.match(r'^[\d+\-*/().\s]+$', question_lower.replace('=', ''))
    if direct_arithmetic:
        try:
            expression = question_lower.split('=')[0].strip()
            result = evaluate_arithmetic(expression)
            return {
                "answer": f"{expression} = {result}",
                "calculation_type": "direct_arithmetic",
                "result": result,
                "success": True
            }
        except ValueError:
            pass
    
    # Mathematical constants and functions
    if "pi" in question_lower:
        return {
            "answer": f"π (Pi) = {math.pi}",
            "calculation_type": "constant",
            "result": math.pi,
            "success": True
        }
    
    if "euler" in question_lower or " e " in question_lower:
        return {
            "answer": f"e (Euler's number) = {math.e}",
            "calculation_type": "constant", 
            "result": math.e,
            "success": True
        }
    
    # If no math pattern found
    return {
        "answer": "No mathematical calculation found in the question",
        "calculation_type": "no_math",
        "result": "not_found",
        "success": False
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "🧮 Math Calculation API",
        "status": "running",
        "available_operations": [
            "Fibonacci numbers and sequences",
            "Basic arithmetic (+, -, *, /)",
            "Mathematical constants (Pi, e)",
            "Expression evaluation"
        ]
    }

@app.post("/calculate", response_model=MathResponse)
async def calculate_math(query: MathQuery):
    """Calculate mathematical expressions"""
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = process_math_question(query.question)
        
        return MathResponse(
            question=query.question,
            answer=result["answer"],
            calculation_type=result["calculation_type"],
            result=result["result"],
            success=result["success"]
        )
    except Exception as e:
        return MathResponse(
            question=query.question,
            answer=f"Error processing question: {str(e)}",
            calculation_type="error",
            result="error",
            success=False
        )

@app.get("/examples")
async def get_examples():
    """Get example mathematical questions"""
    return {
        "fibonacci_examples": [
            "fibonacci 10",
            "10th fibonacci number",
            "fibonacci sequence 8",
            "fib 15"
        ],
        "arithmetic_examples": [
            "5 + 7",
            "12 * 3",
            "what is 15 - 8",
            "calculate 25 / 5",
            "solve 2 + 3 * 4"
        ],
        "constant_examples": [
            "what is pi",
            "value of euler number",
            "pi value"
        ]
    }

@app.get("/test")
async def test_calculations():
    """Test various calculations"""
    test_questions = [
        "fibonacci 5",
        "5 + 7",
        "what is 12 * 3",
        "fibonacci sequence 6",
        "calculate 100 / 4",
        "what is pi"
    ]
    
    results = []
    for question in test_questions:
        result = process_math_question(question)
        results.append({
            "question": question,
            "answer": result["answer"],
            "success": result["success"]
        })
    
    return {"test_results": results}

if __name__ == "__main__":
    print("🧮 Starting Math API Server...")
    print("📖 API Documentation: http://localhost:8002/docs")
    print("🧪 Test endpoint: http://localhost:8002/test")
    print("📚 Examples: http://localhost:8002/examples")
    
    uvicorn.run(
        "math_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )