import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import get_rag_chain

def test_rag_pipeline():
    print("🚀 Starting RAG Pipeline Hallucination Stress Test...\n")
    chain = get_rag_chain()
    
    # Test 1: A valid question (Should answer using context)
    question_1 = "What are the transparency obligations for general-purpose AI models?"
    print("="*70)
    print(f"📝 TEST 1 (Valid Question): {question_1}")
    print("-"*70)
    response_1 = chain.invoke(question_1)
    print(f"🤖 VeriDoc Answer:\n{response_1.content}")
    print("="*70 + "\n")
    
    # Test 2: A trap question (Should REFUSE to answer)
    question_2 = "Who won the FIFA World Cup in 2022 and what is the best recipe for pasta?"
    print("="*70)
    print(f"🪤 TEST 2 (Trap Question): {question_2}")
    print("-"*70)
    response_2 = chain.invoke(question_2)
    print(f"🤖 VeriDoc Answer:\n{response_2.content}")
    print("="*70 + "\n")
    
    # Evaluation Check
    if "cannot answer" in response_2.content.lower() or "not explicitly stated" in response_2.content.lower():
        print("🎉 SUCCESS: The LLM successfully resisted hallucination and refused to answer the trap question!")
    else:
        print("⚠️ WARNING: The LLM hallucinated an answer instead of refusing. Check the prompt.")

if __name__ == "__main__":
    test_rag_pipeline()