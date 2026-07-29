import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config_loader import get_config
from app.agents import run_agentic_rag, index  # Load the retriever index
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Define the golden evaluation questions and ground truths matching Chapter 1-5 PDFs
GOLDEN_DATASET = [
    {
        "question": "What is a decomposition reaction? Give an example.",
        "ground_truth": "A decomposition reaction is a chemical reaction in which a single reactant breaks down into two or more simpler products. An example is the thermal decomposition of calcium carbonate (limestone) into calcium oxide (quicklime) and carbon dioxide when heated: CaCO3 (s) --heat--> CaO (s) + CO2 (g)."
    },
    {
        "question": "What is the pH scale, and what does a pH of 7 represent?",
        "ground_truth": "A pH scale is a scale for measuring hydrogen ion concentration in a solution, ranging from 0 (very acidic) to 14 (very alkaline). A pH value of 7 represents a neutral solution, such as pure water. Values less than 7 represent an acidic solution, while values greater than 7 represent a basic solution."
    },
    {
        "question": "What is the reactivity series of metals, and how is it used?",
        "ground_truth": "The reactivity series is a list of metals arranged in the order of their decreasing chemical activities. Metals at the top of the series (such as potassium, sodium, calcium) are highly reactive and easily displace metals below them (such as copper, iron) from their salt solutions, which helps in displacement reaction analysis and metal extraction."
    },
    {
        "question": "Why does carbon form covalent bonds and not ionic bonds?",
        "ground_truth": "Carbon has an atomic number of 6 with 4 valence electrons. It cannot gain 4 electrons to form C4- because it is difficult for a nucleus with 6 protons to hold 10 electrons. It cannot lose 4 electrons to form C4+ because removing 4 electrons requires a large amount of energy. Therefore, carbon achieves stability by sharing its valence electrons with other atoms, forming covalent bonds."
    },
    {
        "question": "State Mendeleev's Periodic Law and how it differs from the Modern Periodic Law.",
        "ground_truth": "Mendeleev's Periodic Law states that the chemical and physical properties of elements are a periodic function of their atomic masses. In contrast, the Modern Periodic Law states that the properties of elements are a periodic function of their atomic numbers."
    }
]

def run_evaluation():
    config = get_config()
    models_config = config["models"]
    
    if index is None:
        print("Error: Vector Database Index is not initialized. Please run ingestion first.")
        return
        
    print("Preparing evaluation dataset...")
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []
    
    retriever = index.as_retriever(similarity_top_k=3)
    
    for idx, item in enumerate(GOLDEN_DATASET):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{idx+1}/{len(GOLDEN_DATASET)}] Querying: '{q}'...")
        
        # 1. Retrieve raw contexts
        nodes = retriever.retrieve(q)
        contexts = [node.node.get_content() for node in nodes]
        
        # 2. Get answer from Agentic RAG
        try:
            ans = run_agentic_rag(q)
        except Exception as e:
            print(f"Error querying Agentic RAG: {e}")
            ans = "Error generating answer."
            
        questions.append(q)
        contexts_list.append(contexts)
        answers.append(ans)
        ground_truths.append(gt)
        
    # Construct HuggingFace Dataset
    dataset_dict = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print("Initializing RAGAs evaluators using local Ollama model...")
    # Wrap Ollama model for LangChain (which RAGAs uses)
    evaluator_llm = ChatOllama(
        model=models_config["llm"], 
        base_url=models_config["ollama_url"],
        timeout=180
    )
    evaluator_embeddings = OllamaEmbeddings(
        model=models_config["embedding"],
        base_url=models_config["ollama_url"]
    )
    
    print("Running RAGAs evaluation... (this might take a few minutes)")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevance, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        # Convert output to a dictionary and format print
        output_data = {
            "timestamp": time.time(),
            "global_scores": dict(result),
            "raw_dataset": dataset_dict
        }
        
        # Print results
        print("\n================ EVALUATION SUMMARY ================")
        for metric, score in output_data["global_scores"].items():
            print(f"{metric.capitalize()}: {score:.4f}")
        print("====================================================")
        
        # Save to JSON
        output_path = Path(config["paths"]["pdf_dir"]).parent / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, default=str)
        print(f"Detailed evaluation results saved to: {output_path}")
        
        return output_data
    except Exception as e:
        print(f"RAGAs evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_evaluation()
