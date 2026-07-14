import os
import warnings
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Silence warnings for clean output
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import posthog
posthog.capture = lambda *args, **kwargs: None

def build_persistent_vector_store(pdf_path: str, persist_directory: str = "./chroma_db"):
    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    
    print("🧠 Initializing Embeddings & Semantic Chunker...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    print("🧬 Generating Semantic Chunks...")
    chunks = semantic_splitter.split_text(full_text)
    print(f"   -> Generated {len(chunks)} semantic chunks.")
    
    print(f"💾 Initializing Persistent ChromaDB at: {persist_directory}")
    # Use PersistentClient to save vectors to disk
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get the collection
    collection_name = "eu_ai_act_semantic"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Check if we already have data to avoid re-embedding
    if collection.count() > 0:
        print(f"   -> Collection already contains {collection.count()} documents. Skipping embedding.")
        return collection
    
    print("🚀 Embedding and storing chunks in ChromaDB (this takes ~1-2 mins)...")
    
    # ChromaDB requires unique IDs for each document
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add documents and their embeddings to the collection
    collection.add(
        documents=chunks,
        ids=ids
        # Note: Langchain's SemanticChunker doesn't return metadata by default in split_text, 
        # but we could add page numbers if we used split_documents instead.
    )
    
    print(f"✅ Successfully stored {collection.count()} chunks in ChromaDB!")
    return collection

if __name__ == "__ main__":
    pdf_path = "data/eu_ai_act.pdf"
    if os.path.exists(pdf_path):
        build_persistent_vector_store(pdf_path)
    else:
        print("❌ PDF not found. Please ensure data/eu_ai_act.pdf exists.")