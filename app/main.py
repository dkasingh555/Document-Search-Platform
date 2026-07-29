import time
import uuid
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config_loader import get_config
from app.tracing import init_tracing
from app.ingest import run_ingestion
from app.agents import run_agentic_rag

# Initialize Arize Phoenix tracing
init_tracing()

app = FastAPI(
    title="Document Search Platform Agentic RAG API",
    description="REST API backend for integrating with OpenWebUI, built with Docling, PGVector, LlamaIndex, CrewAI, and Arize Phoenix.",
    version="1.0.0"
)

# Configuration settings
config = get_config()

# Data transfer schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    result: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# Global state to track background ingestion progress
ingestion_state = {
    "status": "idle",
    "message": "Ingestion has not been run yet.",
    "last_run": None
}

def run_background_ingestion():
    global ingestion_state
    ingestion_state["status"] = "running"
    ingestion_state["message"] = "Document ingestion in progress..."
    try:
        success = run_ingestion()
        if success:
            ingestion_state["status"] = "success"
            ingestion_state["message"] = "Ingestion completed successfully."
        else:
            ingestion_state["status"] = "failed"
            ingestion_state["message"] = "Ingestion failed during processing. Check backend logs."
    except Exception as e:
        ingestion_state["status"] = "error"
        ingestion_state["message"] = f"Ingestion failed with exception: {str(e)}"
    ingestion_state["last_run"] = time.time()

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Direct query endpoint to run Agentic RAG."""
    try:
        result = run_agentic_rag(request.query)
        return QueryResponse(query=request.query, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible completion endpoint, allowing direct connection from OpenWebUI."""
    user_messages = [msg.content for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in the chat history.")
    
    # Extract the latest user message
    query = user_messages[-1]
    
    try:
        # Run agentic query
        result = run_agentic_rag(query)
        
        chat_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": result
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    """Trigger PDF ingestion in a background task."""
    global ingestion_state
    if ingestion_state["status"] == "running":
        return {"message": "Ingestion is already running.", "state": ingestion_state}
    
    background_tasks.add_task(run_background_ingestion)
    return {"message": "Ingestion started in the background.", "state": ingestion_state}

@app.get("/api/v1/ingest/status")
def get_ingest_status():
    """Retrieve the status of the background ingestion process."""
    return ingestion_state

@app.post("/api/v1/evaluate")
def run_evaluation_endpoint(background_tasks: BackgroundTasks):
    """Triggers the pipeline evaluation in the background."""
    try:
        from app.evaluate import run_evaluation
        # We will implement this script next
        background_tasks.add_task(run_evaluation)
        return {"message": "RAG evaluation triggered in the background. Check logs or check results JSON later."}
    except ImportError:
        raise HTTPException(status_code=501, detail="Evaluation module is still being implemented.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
