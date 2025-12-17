#!/usr/bin/env python3
"""
Factorization API - Find factors of numbers
API for finding factors of numbers like 9, 46, 64
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import re

app = FastAPI(
    title="🔢 Factorization API",
    description="Find factors of numbers API",
    version="1.0.0"
)

class FactorQuery(BaseModel):
    question: str

class FactorResponse(BaseModel):
    question: str
    answer: str
    number: int
    factors: List[int]
    success: bool

def find_factors(number: int) -> List[int]:
    """Find all factors of a given number"""
    factors = []
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            factors.append(i)
            if i != number // i:  # Avoid duplicates for perfect squares
                factors.append(number // i)
    return sorted(factors)

def process_factorization(expression: str) -> dict:
    """Process factorization requests for numbers like 9, 46, 64"""
    expression_clean = expression.replace(" ", "").lower()
    
    # Match factorization patterns
    factor_patterns = [
        r'^(\d+)$',                                    # Just a number: 9, 46, 64
        r'factor\s*of\s*(\d+)',                       # factor of 9
        r'factors\s*of\s*(\d+)',                      # factors of 46
        r'what\s*is\s*factors?\s*of\s*(\d+)',         # what is factor of 9
        r'what\s*are\s*the\s*factors?\s*of\s*(\d+)',  # what are the factors of 9
        r'find\s*factors?\s*of\s*(\d+)',              # find factors of 64
        r'(\d+)\s*factors?',                          # 9 factors
        r'factorize\s*(\d+)',                         # factorize 46
        r'prime\s*factors?\s*of\s*(\d+)',             # prime factors of 64
    ]
    
    for pattern in factor_patterns:
        match = re.search(pattern, expression_clean)
        if match:
            try:
                number = int(match.group(1))
                if number <= 0:
                    continue
                    
                factors = find_factors(number)
                
                return {
                    "answer": f"Factors of {number}: {factors}",
                    "number": number,
                    "factors": factors,
                    "success": True
                }
            except (ValueError, AttributeError):
                continue
    
    # Special cases for specific numbers
    special_numbers = {
        "9": 9,
        "46": 46, 
        "64": 64,
        "nine": 9,
        "fortysix": 46,
        "sixtyfour": 64
    }
    
    for word, num in special_numbers.items():
        if word in expression_clean:
            factors = find_factors(num)
            return {
                "answer": f"Factors of {num}: {factors}",
                "number": num,
                "factors": factors,
                "success": True
            }
    
    return {
        "answer": "Not a valid factorization request",
        "number": 0,
        "factors": [],
        "success": False
    }

@app.get("/")
async def root():
    return {
        "message": "🔢 Factorization API",
        "status": "running",
        "examples": ["9", "factors of 46", "find factors of 64", "factorize 25"]
    }

@app.post("/factors", response_model=FactorResponse)
async def get_factors(query: FactorQuery):
    """Find factors of a number"""
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = process_factorization(query.question)
        
        return FactorResponse(
            question=query.question,
            answer=result["answer"],
            number=result["number"],
            factors=result["factors"],
            success=result["success"]
        )
    except Exception as e:
        return FactorResponse(
            question=query.question,
            answer=f"Error processing factorization: {str(e)}",
            number=0,
            factors=[],
            success=False
        )

@app.get("/test")
async def test_factorization():
    """Test various factorization operations"""
    test_questions = [
        "9",
        "factors of 46", 
        "find factors of 64",
        "factorize 25",
        "what are the factors of 36"
    ]
    
    results = []
    for question in test_questions:
        result = process_factorization(question)
        results.append({
            "question": question,
            "answer": result["answer"],
            "success": result["success"]
        })
    
    return {"test_results": results}

if __name__ == "__main__":
    print("🔢 Starting Factorization API Server...")
    print("📖 API Documentation: http://localhost:8003/docs")
    print("🧪 Test endpoint: http://localhost:8003/test")
    
    uvicorn.run(
        "factorization_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )