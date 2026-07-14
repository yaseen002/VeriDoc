import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Silence warnings for clean output
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_and_chunk_pdf(pdf_path: str):
    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   -> Loaded {len(pages)} pages.")
    
    # Combine all page text into one giant string for semantic chunking
    full_text = "\n".join([page.page_content for page in pages])
    
    # Initialize Embeddings
    print("🧠 Initializing Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # --- STRATEGY 1: Recursive Character Splitting (Baseline) ---
    print("\n⚙️ Applying Recursive Character Splitting...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # ~200 words
        chunk_overlap=100,    # Keep 100 chars overlap to preserve context
        separators=["\n\n", "\n", ".", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(full_text)
    print(f"   -> Generated {len(recursive_chunks)} chunks.")
    
    # --- STRATEGY 2: Semantic Chunking (Advanced) ---
    print("\n🧬 Applying Semantic Chunking (This may take a minute for large PDFs)...")
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95  # Break when similarity drops below 95th percentile
    )
    semantic_chunks = semantic_splitter.split_text(full_text)
    print(f"   -> Generated {len(semantic_chunks)} chunks.")
    
    return recursive_chunks, semantic_chunks

if __name__ == "__main__":
    pdf_path = "data/eu_ai_act.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found. Please download the EU AI Act PDF and place it in the 'data/' folder.")
    else:
        rec_chunks, sem_chunks = load_and_chunk_pdf(pdf_path)
        print("\n✅ Ingestion and Chunking Complete!")