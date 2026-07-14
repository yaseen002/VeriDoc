import streamlit as st
import sys
import os

# Add root to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Silence warnings and telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import posthog
posthog.capture = lambda *args, **kwargs: None
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="VeriDoc", page_icon="📜", layout="wide")

st.title("📜 VeriDoc")
st.markdown("#### *Domain-Specific RAG System | Strictly grounded in the EU AI Act*")

# --- CACHED INITIALIZATION ---
@st.cache_resource
def initialize_components():
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0, api_key=os.getenv("MISTRAL_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="eu_ai_act_semantic")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """You are VeriDoc, a strict, factual, and highly accurate regulatory AI assistant. 
Your task is to answer the user's question based *ONLY* on the provided context from the EU AI Act.
<context>
{context}
</context>
Rules:
1. If the answer is not explicitly stated in the <context>, you MUST reply with exactly: "I cannot answer this question based on the provided EU AI Act documents."
2. Do not use any outside knowledge.
3. Be concise and cite the specific Article if mentioned.
Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)
    return chain, retriever

chain, retriever = initialize_components()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []

# --- TRANSPARENCY SIDEBAR ---
with st.sidebar:
    st.header("🔍 Transparency Panel")
    st.info("VeriDoc only answers based on the retrieved context. If the answer isn't in the documents, it will explicitly refuse to answer.")
    
    if st.session_state.last_contexts:
        st.subheader("📄 Retrieved Context")
        for i, ctx in enumerate(st.session_state.last_contexts):
            with st.expander(f"Chunk {i+1}"):
                st.text(ctx[:600] + "...")
    else:
        st.info("Ask a question to see the retrieved context here.")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Ask a question about the EU AI Act..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching the EU AI Act and generating grounded response..."):
            # 1. Retrieve context
            docs = retriever.invoke(prompt)
            contexts = [doc.page_content for doc in docs]
            st.session_state.last_contexts = contexts # Update sidebar
            
            # 2. Generate answer
            response = chain.invoke(prompt)
            answer = response.content
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})