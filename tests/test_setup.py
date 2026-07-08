import os
import warnings

# 1. Silence ChromaDB telemetry warnings BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 2. Silence LangChain deprecation warnings for HuggingFaceEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings # Reverted to community
from langchain_core.messages import HumanMessage
import chromadb
from dotenv import load_dotenv

load_dotenv()

def test_infrastructure():
    print("1. Testing Mistral API (Cloud LLM)...")
    llm = ChatMistralAI(
        model="mistral-small-latest", 
        temperature=0, 
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    message = HumanMessage(content="Say 'VeriDoc API infrastructure is ready!' in exactly those words.")
    response = llm.invoke([message])
    print(f"   -> LLM Response: {response.content.strip()}")

    print("\n2. Testing Embeddings (SentenceTransformers)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    test_vector = embeddings.embed_query("This is a test sentence.")
    print(f"   -> Embedding dimension: {len(test_vector)}") 

    print("\n3. Testing Vector Database (ChromaDB)...")
    client = chromadb.Client()
    collection = client.create_collection(name="test_collection")
    collection.add(
        documents=["VeriDoc is awesome."],
        embeddings=[test_vector],
        ids=["id1"]
    )
    results = collection.query(query_embeddings=[test_vector], n_results=1)
    print(f"   -> ChromaDB Query Result: {results['documents'][0][0]}")

    print("\n✅ Phase 1 Setup Complete! API, Embeddings, and Vector DB are all connected.")

if __name__ == "__main__":
    test_infrastructure()