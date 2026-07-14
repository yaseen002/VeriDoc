import sys
import os

# Add the root directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_and_chunk_pdf

def test_chunking_strategies():
    pdf_path = "data/eu_ai_act.pdf"
    
    if not os.path.exists(pdf_path):
        print("❌ PDF not found. Skipping test.")
        return

    print("🔄 Running chunking comparison... (This might take a minute)\n")
    rec_chunks, sem_chunks = load_and_chunk_pdf(pdf_path)
    
    print("\n" + "="*50)
    print("🔍 COMPARISON: CHUNK #5 FROM BOTH METHODS")
    print("="*50)
    
    # Let's look at the 5th chunk (index 4) to see how they differ
    print("\n📏 RECURSIVE CHUNK (Strict character limits):")
    print("-" * 30)
    if len(rec_chunks) > 4:
        print(rec_chunks[4][:500] + "...") # Print first 500 chars
        print(f"[Total length: {len(rec_chunks[4])} characters]")
    else:
        print("Not enough chunks generated.")

    print("\n🧬 SEMANTIC CHUNK (Grouped by meaning):")
    print("-" * 30)
    if len(sem_chunks) > 4:
        print(sem_chunks[4][:500] + "...") # Print first 500 chars
        print(f"[Total length: {len(sem_chunks[4])} characters]")
    else:
        print("Not enough chunks generated.")

    print("\n" + "="*50)
    print("💡 OBSERVATION:")
    print("Notice how the Recursive chunk might cut off mid-sentence or combine unrelated paragraphs just to hit the 800-character limit.")
    print("Notice how the Semantic chunk contains a complete, unified thought because it only breaks when the topic changes!")
    print("="*50)

if __name__ == "__main__":
    test_chunking_strategies()