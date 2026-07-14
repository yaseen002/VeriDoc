import os
import warnings
from dotenv import load_dotenv

# LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Silence warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    print("🔌 Initializing RAG Pipeline...")
    
    # 1. Initialize LLM (Strict, low temperature for factual accuracy)
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.0,  # 0.0 ensures deterministic, factual outputs
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    # 2. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Connect to Persistent ChromaDB
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="eu_ai_act_semantic"
    )
    
    # 4. Create Retriever (Fetch top 3 most relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. Define the STRICT System Prompt (The anti-hallucination shield)
    prompt_template = """You are VeriDoc, a strict, factual, and highly accurate regulatory AI assistant. 
Your task is to answer the user's question based *ONLY* on the provided context from the EU AI Act.

<context>
{context}
</context>

Rules:
1. If the answer is not explicitly stated in the <context>, you MUST reply with exactly: "I cannot answer this question based on the provided EU AI Act documents."
2. Do not use any outside knowledge or make assumptions.
3. Be concise, professional, and cite the specific Article or Recital number if it is mentioned in the context.

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 6. Build the LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    print("✅ RAG Pipeline initialized successfully!")
    return rag_chain

if __name__ == "__main__":
    # Quick test to ensure it loads
    chain = get_rag_chain()
    print("Pipeline ready for queries.")