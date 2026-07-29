import os
from llama_index.core import set_global_handler

def init_tracing():
    # Point OpenTelemetry to our local Arize Phoenix docker container
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
    os.environ["OTEL_SERVICE_NAME"] = "agentic-rag-backend"
    
    # 1. Instrument LlamaIndex
    try:
        set_global_handler("arize_phoenix")
        print("LlamaIndex tracing initialized via Arize Phoenix.")
    except Exception as e:
        print(f"Failed to initialize LlamaIndex tracing: {e}")
        
    # 2. Instrument CrewAI (which uses LangChain under the hood)
    try:
        from phoenix.trace.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        print("CrewAI/LangChain tracing initialized via Arize Phoenix.")
    except Exception as e:
        print(f"Failed to initialize CrewAI/LangChain tracing: {e}")

if __name__ == "__main__":
    init_tracing()
