import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import build_persistent_vector_store

def test_persistent_db():
    pdf_path = "data/eu_ai_act.pdf"
    
    if not os.path.exists(pdf_path):
        print("❌ PDF not found. Skipping test.")
        return

    print("🔄 Building / Loading Persistent Vector Store...\n")
    collection = build_persistent_vector_store(pdf_path)
    
    print("\n🔍 Testing Similarity Search...")
    # Let's search for something specific in the EU AI Act
    query = "What are the requirements for high-risk AI systems regarding data governance?"
    
    results = collection.query(
        query_texts=[query],
        n_results=2 # Retrieve top 2 most relevant chunks
    )
    
    print(f"\n📌 Query: '{query}'")
    print("-" * 60)
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n[Result {i+1}] (ID: {results['ids'][0][i]})")
        print(doc[:400] + "...\n") # Print first 400 chars of each result

    print("✅ Vector Store Test Complete!")

if __name__ == "__main__":
    test_persistent_db()