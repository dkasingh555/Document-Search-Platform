# Presentation: Document Search Platform with Agentic RAG

---

## Slide 1: Title
### **Document Search Platform with Agentic RAG**
*Exposing REST APIs for OpenWebUI Integration*

**Built with:**
- **Docling** for Document Preprocessing
- **PostgreSQL & PGVector** as the Vector Database
- **LlamaIndex** for semantic chunking & indexing
- **CrewAI** for multi-agent reasoning & retrieval
- **Ollama (llama3.1 & nomic-embed-text)** for local inference
- **Arize Phoenix** for OpenTelemetry tracing & observability
- **RAGAs** for contextual evaluation

---

## Slide 2: The Challenge
### **Textbook Q&A & Search Requirements**
- Target: Grade 10 CBSE Science Textbooks (Chapters 1 to 5: Chemistry).
- Challenge:
  - Standard PDF parsers lose formatting (headers, lists, tables).
  - Traditional RAG (direct vector search + prompt) lacks critical reasoning and fails on complex or multi-part questions.
  - Absence of tracing makes debugging and optimizing retrieval/generation difficult.
  - Quantitative evaluation of RAG quality (faithfulness, precision) is rarely standardized.

---

## Slide 3: Solution Architecture
### **System Layout & Data Flow**
- **Frontend**: OpenWebUI (Port 3000) provides the chat interface. Connects to custom backend via OpenAI protocol.
- **REST API Gateway**: FastAPI (Port 8000) exposes query, completions, and ingestion status.
- **Agentic RAG Engine**: CrewAI coordinates collaborative retrieval. LlamaIndex performs vector search via PGVector.
- **Storage**: Postgres (Port 5432) with PGVector stores text nodes and 768-dimensional embeddings.
- **Observability**: Arize Phoenix (Port 6006) captures OTel trace telemetry.
- **Evaluation**: RAGAs evaluates metrics using local Ollama instance.

---

## Slide 4: Ingestion Pipeline
### **Docling + Hierarchical Splitting**
1. **Preprocessing (Docling)**:
   - Parses Chemistry PDFs and outputs structured Markdown.
   - Retains layout structure, lists, and tables.
2. **Hierarchical Splitting**:
   - LlamaIndex `MarkdownNodeParser` splits document content along header boundaries (`#`, `##`, `###`).
   - Keeps headers with paragraphs, improving context retention.
3. **Indexing**:
   - Embeds nodes using `nomic-embed-text` (768 dimensions).
   - Stores node text, metadata (filename, path), and vector embeddings in PostgreSQL using `PGVectorStore`.

---

## Slide 5: Multi-Agent Retrieval (CrewAI)
### **Reasoning & Synthesis Agents**
Rather than direct LLM completion, the system uses two cooperating CrewAI agents:
1. **Textbook Retriever Agent**:
   - Goal: Search and retrieve relevant context.
   - Tool: `Science Textbook Search Tool` wraps LlamaIndex vector retriever.
2. **Science Expert Teacher Agent**:
   - Goal: Analyze search results, cross-verify details, and formulate a clear, pedagogically sound response.
   - Backstory: A master tutor who ensures answers are factually grounded and avoids speculation.

- Tasks are defined in external `prompts/prompts.yaml`, keeping system prompt management independent of application code.

---

## Slide 6: Tracing & Observability
### **OpenTelemetry + Arize Phoenix**
- The FastAPI server, LlamaIndex, and CrewAI are instrumented with OpenTelemetry.
- Traces are exported via gRPC to Arize Phoenix (running in Docker, Port 6006).
- **What is traced**:
  - API HTTP requests.
  - CrewAI task execution and agent prompts.
  - Tool calls (`Science Textbook Search Tool`).
  - LlamaIndex PGVector query performance and returned nodes.
  - Ollama LLM generation tokens and latency.

---

## Slide 7: RAGAs Evaluation Framework
### **System Performance Metrics**
We evaluate the RAG pipeline using a golden set of Grade 10 Science questions (Chemical Reactions, Acids/Bases, Metals, Carbon, Periodic Classification) against:
- **Faithfulness**: Validates that generated answers contain only information present in the context.
- **Answer Relevance**: Checks if the answer matches the query intent.
- **Context Precision**: Measures how well the retriever ranks relevant context chunks at the top.
- **Context Recall**: Verifies if the retrieved context contains all parts of the ground-truth answer.

*Evaluation runs locally using the Ollama backend and saves results to `evaluation_results.json`.*

---

## Slide 8: Deployment & Getting Started
### **Setup in 3 Steps**
1. **Start Infrastructure**:
   ```bash
   docker compose up -d
   ```
   *Spins up PostgreSQL (PGVector), Arize Phoenix, and OpenWebUI.*
2. **Ingest Documents**:
   ```bash
   python app/ingest.py
   # Or send POST request to http://localhost:8000/api/v1/ingest
   ```
3. **Run API & Chat**:
   - Start FastAPI: `uvicorn app.main:app --reload`
   - Open OpenWebUI (`http://localhost:3000`), add the custom API connection (`http://host.docker.internal:8000/v1`), and start searching textbooks!
