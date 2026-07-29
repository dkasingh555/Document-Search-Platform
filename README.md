# Document Search Platform with Agentic RAG Backend

This repository contains the complete implementation of a **Document Search Platform** featuring an **Agentic RAG backend** integrated with an **OpenWebUI frontend**. 

The backend ingests Grade 10 CBSE Science textbooks (Chemistry Chapters 1 to 5) and exposes REST APIs (including an OpenAI-compatible endpoint) to enable chat-based search and reasoning.

---

## Technical Architecture & Core Features

- **Document Preprocessing**: Uses **Docling** to parse science textbooks from PDF to structured Markdown, preserving lists, headers, and tables.
- **Hierarchical Chunking**: Uses LlamaIndex `MarkdownNodeParser` to split content along header boundaries to keep chapter sections and sub-headings structurally intact.
- **Vector Database**: **PostgreSQL** with the **PGVector** extension (running in Docker) serves as the high-performance vector store.
- **Agentic Retrieval**: Built with **CrewAI** and **LlamaIndex**. A multi-agent team collaborating:
  - *Retriever Agent*: Specializes in querying the vector index using LlamaIndex tool.
  - *Expert Teacher Agent*: Synthesizes, verifies, and drafts educational answers using local `llama3.1:8b` via Ollama.
- **Inference Tracing**: Direct OpenTelemetry instrumentation connected to **Arize Phoenix** for visual analysis of LLM prompts, tool executions, vector queries, and latency.
- **Validation Suite**: **RAGAs** metrics (Faithfulness, Answer Relevance, Context Precision, and Context Recall) evaluating the system locally.
- **Externalized Prompts**: System prompts, agent roles, goals, and tasks are defined in `prompts/prompts.yaml`, keeping prompt engineering decoupled from Python code.

Supplementary design documents can be found in the `/doc` directory:
- [Solution Architecture and Design Diagrams](doc/architecture.md)
- [Presentation Slide Deck Details](doc/presentation.md)
- [REST API OpenAPI Specification](doc/openapi.json)

---

## Getting Started

### Prerequisites
- **Python**: Version `3.12` installed.
- **Docker Desktop**: Installed and running (ensure WSL 2 backend is active).
- **Ollama**: Installed and running natively on Windows (for GPU acceleration).

---

### Step 1: Start Docker Infrastructure
Launch the database, tracing console, and OpenWebUI using Docker Compose:
```bash
docker compose up -d
```
Verify the containers are running:
- **PostgreSQL (PGVector)**: Port `5432`
- **Arize Phoenix (Trace UI)**: Port `6006` (Open in browser: `http://localhost:6006`)
- **OpenWebUI**: Port `3000` (Open in browser: `http://localhost:3000`)

---

### Step 2: Set Up Python Environment & Download Models
1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Pull the required models in Ollama:
   ```bash
   # LLM for generation/reasoning
   ollama pull llama3.1:8b
   
   # Embedding model for vector indexing
   ollama pull nomic-embed-text
   ```

---

### Step 3: Run Document Ingestion
Ingest the PDF textbooks located in the `pdf/Science X (eng)` directory. The script will parse the PDFs with Docling, generate Markdown nodes, compute embeddings, and insert them into PostgreSQL:
```bash
python app/ingest.py
```
*Note: Ingestion can also be triggered via the REST API endpoint `POST /api/v1/ingest` once the FastAPI server is running.*

---

### Step 4: Run the FastAPI Backend
Start the FastAPI application:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started:
- Access the interactive REST API Swagger UI at: `http://127.0.0.1:8000/docs`
- Complete OpenAPI description is accessible at: `http://127.0.0.1:8000/openapi.json`

---

### Step 5: Integrate with OpenWebUI
1. Open the OpenWebUI interface in your browser at `http://localhost:3000`.
2. Disable login or register a local developer account.
3. Navigate to **Admin Panel > Settings > Connections > OpenAI API**.
4. Configure the connection parameters:
   - **API URL**: `http://host.docker.internal:8000/v1` (points to the FastAPI backend)
   - **API Key**: `dummy` (any placeholder value)
5. Save the configuration.
6. Start a new chat, select the model (`llama3.1:8b`), and ask questions about the Chemistry textbook chapters (e.g., *"What is a decomposition reaction?"*, *"How does Mendeleev's periodic table differ from the modern one?"*).

---

## Observability & Inference Tracing
Every chat query, tool call, database lookup, and LLM invocation is fully traced. 
To inspect the internal trace spans:
1. Open `http://localhost:6006` in your browser.
2. In the **Phoenix UI**, inspect the active spans under the `agentic-rag-backend` project.
3. Expand traces to see the exact input prompts, retrieved documents, model tokens, and execution times.

---

## Evaluating the RAG Pipeline (RAGAs)
To evaluate the accuracy of the Agentic RAG pipeline, run the RAGAs suite:
```bash
python app/evaluate.py
```
*Alternatively, trigger evaluation by sending a POST request to `http://localhost:8000/api/v1/evaluate`.*

This script evaluates:
1. **Faithfulness** (checking for hallucinations)
2. **Answer Relevance**
3. **Context Precision**
4. **Context Recall**

The results are output to the terminal and stored as a JSON report in `pdf/evaluation_results.json`.
