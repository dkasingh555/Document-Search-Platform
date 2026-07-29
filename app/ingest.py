import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config_loader import get_config
from app.db_setup import setup_database

from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings

def run_ingestion():
    # 1. Initialize database & vector extension
    print("Initializing Database...")
    if not setup_database():
        print("Database setup failed!")
        return False
        
    config = get_config()
    db_config = config["database"]
    models_config = config["models"]
    paths_config = config["paths"]
    
    # 2. Configure LlamaIndex global settings
    print("Configuring Embedding model...")
    embed_model = OllamaEmbedding(
        model_name=models_config["embedding"],
        base_url=models_config["ollama_url"],
        request_timeout=120.0
    )
    Settings.embed_model = embed_model
    Settings.llm = None # We will use LLM only in RAG query
    
    # 3. Setup PGVector vector store
    # nomic-embed-text has 768 dimensions
    vector_store = PGVectorStore.from_params(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        table_name="science_x_vectors",
        embed_dim=768
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Load PDFs using Docling
    pdf_dir = Path(paths_config["pdf_dir"])
    if not pdf_dir.exists():
        print(f"PDF directory not found at: {pdf_dir}")
        return False
        
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False
        
    print(f"Found {len(pdf_files)} PDF files to ingest.")
    
    # Initialize Docling Reader
    print("Initializing DoclingReader...")
    reader = DoclingReader()
    
    # Parser for markdown
    node_parser = MarkdownNodeParser()
    
    all_nodes = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        try:
            # Load PDF as LlamaIndex Documents (uses Docling under the hood)
            docs = reader.load_data(file_path=pdf_path)
            print(f"Extracted {len(docs)} document objects from {pdf_path.name}")
            
            # Parse documents into hierarchical markdown nodes
            nodes = node_parser.get_nodes_from_documents(docs)
            print(f"Generated {len(nodes)} markdown nodes for {pdf_path.name}")
            
            # Inject metadata
            for node in nodes:
                node.metadata["file_name"] = pdf_path.name
                node.metadata["file_path"] = str(pdf_path)
                
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            
    if not all_nodes:
        print("No nodes were successfully processed.")
        return False
        
    # 5. Index nodes in Vector Database
    print(f"Indexing {len(all_nodes)} nodes into PGVector... (this will compute embeddings and save them)")
    try:
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True
        )
        print("Ingestion and Indexing completed successfully!")
        return True
    except Exception as e:
        print(f"Failed to create index: {e}")
        return False

if __name__ == "__main__":
    run_ingestion()
