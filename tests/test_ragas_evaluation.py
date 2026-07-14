import os
import sys
import json
import warnings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import posthog
posthog.capture = lambda *args, **kwargs: None

from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# RAGAS imports (Answer Relevancy removed to fix rate limits)
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

def run_ragas_evaluation():
    print("🚀 Starting RAGAS Evaluation...\n")
    
    # 1. Load Ground Truth Dataset
    with open("data/evaluation_dataset.json", "r") as f:
        ground_truth_data = json.load(f)
    
    # 2. Initialize Components
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="eu_ai_act_semantic"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.0,
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # 3. Generate Predictions
    print("🔄 Generating predictions from our RAG pipeline...")
    predictions = []
    
    for item in ground_truth_data:
        question = item["question"]
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        context_str = "\n\n".join(contexts)
        
        prompt = f"""You are VeriDoc, a strict, factual regulatory AI assistant. 
Answer based *ONLY* on this context:
<context>
{context_str}
</context>
If the answer is not in the context, reply: "I cannot answer this question based on the provided EU AI Act documents."
Question: {question}
Answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        predictions.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })
        print(f"   -> Processed: {question[:50]}...")
    
    eval_dataset = Dataset.from_list(predictions)
    
    # 4. Configure RunConfig for Free-Tier Rate Limits
    run_config = RunConfig(
        timeout=1000,
        max_retries=15,
        max_wait=120,
        exception_types=(Exception,)
    )
    
    # 5. Run RAGAS Evaluation (Optimized)
    print("\n📊 Running RAGAS Metrics (Faithfulness, Context Precision)...")
    print("   (Note: Answer Relevancy was dropped to prevent API rate-limit exhaustion)\n")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision], 
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False, 
        run_config=run_config   
    )
    
    # 6. Print Results
    print("="*70)
    print("📈 RAGAS EVALUATION RESULTS")
    print("="*70)
    df = results.to_pandas()
    
    cols_to_show = ['question', 'faithfulness', 'context_precision']
    print(df[cols_to_show].to_string(index=False))
    print("="*70)
    
    avg_faithfulness = df['faithfulness'].mean()
    avg_precision = df['context_precision'].mean()
    
    print(f"\n🏆 AVERAGE SCORES (Perfect = 1.0):")
    print(f"   • Faithfulness:      {avg_faithfulness:.2f} (Measures anti-hallucination)")
    print(f"   • Context Precision: {avg_precision:.2f} (Measures retriever quality)")
    print("="*70)

if __name__ == "__main__":
    run_ragas_evaluation()