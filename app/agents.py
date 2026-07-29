import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config_loader import get_config, get_prompts
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. Initialize configuration
config = get_config()
prompts = get_prompts()
db_config = config["database"]
models_config = config["models"]

# 2. Configure LlamaIndex Embedding (used by vector store / retriever)
embed_model = OllamaEmbedding(
    model_name=models_config["embedding"],
    base_url=models_config["ollama_url"],
    request_timeout=120.0
)
Settings.embed_model = embed_model

# 3. Setup PGVector vector store
vector_store = PGVectorStore.from_params(
    host=db_config["host"],
    port=db_config["port"],
    database=db_config["dbname"],
    user=db_config["user"],
    password=db_config["password"],
    table_name="science_x_vectors",
    embed_dim=768
)

# 4. Load the vector index
try:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    print(f"Warning: Could not connect to PGVector vector store. Ingestion may need to be run first. Error: {e}")
    index = None

# 5. Define LlamaIndex Search Tool for CrewAI
@tool("Science Textbook Search Tool")
def search_textbooks(query: str) -> str:
    """Searches the Science X textbooks database for relevant passages, chapters, concepts, and scientific context matching the query."""
    if index is None:
        return "Error: Database index is not initialized. Please run ingestion first."
    
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        return "No relevant passages found in the textbooks database."
        
    context_str = ""
    for idx, node in enumerate(nodes):
        file_name = node.node.metadata.get("file_name", "Unknown Document")
        context_str += f"[Source {idx+1} - {file_name}]:\n{node.node.get_content()}\n\n"
        
    return context_str

# 6. Initialize LLM for CrewAI
crew_llm = LLM(
    model=f"ollama/{models_config['llm']}",
    base_url=models_config["ollama_url"],
    timeout=180
)

def run_agentic_rag(query: str) -> str:
    # Reload prompts in case they were modified externally
    p_config = get_prompts()
    
    # retriever agent prompts
    retriever_p = p_config["retriever_agent"]
    retriever_agent = Agent(
        role=retriever_p["role"],
        goal=retriever_p["goal"].format(query=query),
        backstory=retriever_p["backstory"],
        tools=[search_textbooks],
        llm=crew_llm,
        verbose=True,
        allow_delegation=False
    )
    
    # analyzer agent prompts
    analyzer_p = p_config["analyzer_agent"]
    analyzer_agent = Agent(
        role=analyzer_p["role"],
        goal=analyzer_p["goal"].format(query=query),
        backstory=analyzer_p["backstory"],
        llm=crew_llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Define tasks
    search_task = Task(
        description=f"Search the database for relevant information regarding the query: '{query}'. Gather all scientific context and facts.",
        expected_output="A compiled text containing retrieved passages and details from textbooks.",
        agent=retriever_agent
    )
    
    task_p = p_config["crew_task"]
    synthesize_task = Task(
        description=task_p["description"].format(query=query),
        expected_output=task_p["expected_output"],
        agent=analyzer_agent
    )
    
    # Assemble crew and run
    crew = Crew(
        agents=[retriever_agent, analyzer_agent],
        tasks=[search_task, synthesize_task],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"Kicking off agentic RAG query for: '{query}'")
    result = crew.kickoff(inputs={"query": query})
    return str(result)

if __name__ == "__main__":
    test_query = "What is gravity and what does the textbook say about gravitational force?"
    res = run_agentic_rag(test_query)
    print("\n--- TEST RESPONSE ---")
    print(res)
