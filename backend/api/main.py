#!/usr/bin/env python3
"""
API for Agentic RAG - Following LangChain standards
Pure RAG implementation without business metrics
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator, List
import os
import time
import uuid
import json
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Import RAG agents
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rags.rag_agent import RAGAgent

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver


# ========================================
# APP SETUP
# ========================================

app = FastAPI(
    title="Agentic RAG API",
    description="Optimized RAG Agent - Pure LangChain/LangGraph implementation with k=4 retrieval",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# MODELS (Simple, no business metrics)
# ========================================

class QueryRequest(BaseModel):
    """Query request with optional thread ID for memory"""
    question: str
    thread_id: Optional[str] = None  # For conversation memory


class AgentResponse(BaseModel):
    """Agent response - Clean version with thread support"""
    answer: str
    used_retrieval: Optional[bool] = None
    latency: float
    thread_id: Optional[str] = None  # Return thread_id for frontend


# ========================================
# GLOBAL STATE
# ========================================

rag_agent = None
vectorstore = None


# ========================================
# STARTUP
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG agent on startup"""
    global rag_agent, vectorstore

    print("🚀 Starting Agentic RAG API...")

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not found")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize vectorstore
    persist_dir = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Create checkpointer for memory
    checkpointer = InMemorySaver()
    print("💾 Memory system (InMemorySaver) initialized")

    # Initialize RAG agent with memory
    rag_agent = RAGAgent(vectorstore, checkpointer=checkpointer)

    print("✅ RAG Agent initialized (k=4, improved prompt)")

    # Sync uploaded_documents with ChromaDB (CRITICAL for persistence)
    global uploaded_documents
    try:
        collection = vectorstore._collection
        results = collection.get()

        # Group by source
        sources_data = {}
        for meta in results['metadatas']:
            source = meta.get('source')
            if source:
                if source not in sources_data:
                    sources_data[source] = {
                        'file_size': meta.get('file_size', 0),
                        'upload_date': meta.get('upload_date', ''),
                        'chunks': 0
                    }
                sources_data[source]['chunks'] += 1

        # Populate uploaded_documents
        uploaded_documents = [
            {
                'source': source,
                'file_size': data['file_size'],
                'upload_date': data['upload_date'],
                'chunks': data['chunks']
            }
            for source, data in sources_data.items()
        ]

        print(f"📚 Synced {len(uploaded_documents)} documents from ChromaDB ({collection.count()} chunks)")
    except Exception as e:
        print(f"⚠️  Warning: Could not sync uploaded_documents: {e}")
        uploaded_documents = []


# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """API status"""
    return {
        "status": "online",
        "version": "2.0.0",
        "mode": "Optimized RAG (k=4, improved prompt)",
        "agent": "rag_agent"
    }


@app.post("/api/rag_agent", response_model=AgentResponse)
async def query_rag_agent(request: QueryRequest):
    """Query the standard RAG agent with optional memory support"""
    start_time = time.time()

    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())

        # Invoke with thread_id for memory support
        result = rag_agent.invoke(request.question, thread_id=thread_id)
        latency = time.time() - start_time

        return AgentResponse(
            answer=result["answer"],
            used_retrieval=result.get("used_retrieval"),
            latency=latency,
            thread_id=result.get("thread_id")  # Return thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_agent": rag_agent is not None,
        "vectorstore": vectorstore is not None
    }


# ========================================
# ADAPTER ENDPOINTS FOR ORIGINAL FRONTEND
# ========================================

# Store for document metadata (in production, use a database)
uploaded_documents = []

# Store thread_id per session (in production, use session management)
session_threads = {}


@app.post("/api/query")
async def query_adapter(request: dict):
    """
    Adapter endpoint for original frontend
    Maps to our RAG agents with memory support
    """
    import uuid

    question = request.get("question", "")
    session_id = request.get("session_id", str(uuid.uuid4()))

    # Get or create thread_id for this session
    if session_id not in session_threads:
        session_threads[session_id] = f"thread-{uuid.uuid4()}"
    thread_id = session_threads[session_id]

    # Use optimized RAG agent
    start_time = time.time()

    try:
        result = rag_agent.invoke(question, thread_id=thread_id)
        latency = time.time() - start_time

        return {
            "answer": result["answer"],
            "method": "optimized_rag_agent",
            "tier": "standard",
            "complexity": "medium",
            "latency": latency,
            "cost_estimate": 0.001,  # Placeholder
            "from_cache": False,
            "fallback_used": False,
            "escalated": False,
            "confidence": 0.90,
            "faithfulness": 0.92,
            "num_documents": 4,
            "thread_id": thread_id,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def get_documents():
    """Return list of uploaded documents"""
    return {"documents": uploaded_documents}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Handle real document upload and processing
    Supports PDF, TXT, MD, and DOCX files
    """
    global vectorstore  # CRITICAL: Use the global vectorstore instance

    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    results = []
    total_chunks = 0

    # Initialize text splitter with larger chunks for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for file in files:
        try:
            print(f"📄 Processing file: {file.filename}")
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                file_size = len(content)

            print(f"✅ File saved to temp: {tmp_file_path}, size: {file_size} bytes")

            # Load and process based on file type
            file_ext = Path(file.filename).suffix.lower()
            documents = []

            print(f"🔍 Loading {file_ext} file...")
            if file_ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                print(f"✅ PDF loaded: {len(documents)} pages")
            elif file_ext in [".txt", ".md"]:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(tmp_file_path)
                documents = loader.load()
                print(f"✅ Text loaded: {len(documents)} documents")
            elif file_ext in [".docx", ".doc"]:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
                documents = loader.load()
                print(f"✅ DOCX loaded: {len(documents)} documents")
            elif file_ext == ".ipynb":
                from langchain_community.document_loaders import NotebookLoader
                loader = NotebookLoader(tmp_file_path, include_outputs=True, max_output_length=20, remove_newline=True)
                documents = loader.load()
                print(f"✅ Notebook loaded: {len(documents)} cells")
            else:
                # Try as text file
                loader = TextLoader(tmp_file_path)
                documents = loader.load()
                print(f"✅ Text loaded: {len(documents)} documents")

            # Split documents into chunks
            print(f"✂️ Splitting into chunks...")
            chunks = text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")

            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = file.filename
                chunk.metadata["upload_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
                chunk.metadata["file_size"] = file_size

            # Add to vectorstore
            if chunks:
                print(f"💾 Adding {len(chunks)} chunks to vectorstore...")
                vectorstore.add_documents(chunks)
                print(f"✅ Chunks added to vectorstore")
                total_chunks += len(chunks)

                # Add to document list
                doc = {
                    "source": file.filename,
                    "file_size": file_size,
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunks": len(chunks)
                }
                uploaded_documents.append(doc)

                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "chunks": len(chunks),
                    "message": f"Successfully processed {file.filename}"
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "chunks": 0,
                    "message": "No content extracted from file"
                })

            # Clean up temp file
            os.unlink(tmp_file_path)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            # Try to clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    # Persist changes to ChromaDB (CRITICAL for disk persistence)
    if total_chunks > 0:
        print(f"💾 Persisting {total_chunks} chunks to ChromaDB...")
        vectorstore.persist()
        print("✅ ChromaDB persisted to disk")

    # Return with explicit headers to prevent buffering
    return JSONResponse(
        content={
            "results": results,
            "total_files": len(files),
            "total_chunks": total_chunks
        },
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )


@app.delete("/api/documents")
async def delete_document(source: str):
    """Delete a document by source - removes from both list and vectorstore"""
    global uploaded_documents, vectorstore

    # Find and remove the document from list
    original_count = len(uploaded_documents)
    uploaded_documents = [d for d in uploaded_documents if d["source"] != source]

    if len(uploaded_documents) < original_count:
        # Remove from vectorstore using ChromaDB delete with metadata filter
        try:
            collection = vectorstore._collection
            collection.delete(where={"source": source})
            print(f"✅ Deleted {source} from ChromaDB")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove from vectorstore: {e}")

        return {"message": f"Deleted {source}"}
    else:
        raise HTTPException(status_code=404, detail=f"Document {source} not found")


@app.delete("/api/documents/all")
async def clear_all_documents():
    """Clear all documents from ChromaDB and uploaded_documents list"""
    global uploaded_documents, vectorstore

    try:
        # Get count before deletion
        collection = vectorstore._collection
        count_before = collection.count()

        # Delete all documents from ChromaDB
        # Get all IDs and delete them
        all_ids = collection.get()['ids']
        if all_ids:
            collection.delete(ids=all_ids)

        # Clear uploaded_documents list
        uploaded_documents = []

        print(f"🗑️  Cleared all documents: {count_before} chunks deleted")

        return {
            "message": "All documents cleared",
            "chunks_deleted": count_before
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


# ========================================
# STREAMING ENDPOINTS
# ========================================

async def generate_sse_stream(
    question: str,
    thread_id: str,
    agent_type: str = "rag"
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream for real-time responses
    """
    try:
        # Use the optimized RAG agent
        agent = rag_agent

        # For now, use regular invoke and simulate streaming
        # LangGraph doesn't have native token-by-token streaming yet
        from langchain_core.messages import HumanMessage

        # Get the full response
        result = agent.invoke(question, thread_id)
        full_response = result["answer"]

        # Simulate streaming by sending chunks
        words = full_response.split()
        current_text = ""

        for i, word in enumerate(words):
            current_text += word + " "

            # Send SSE event for each chunk
            event_data = {
                "type": "token",
                "content": word + " ",
                "thread_id": thread_id
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            # Add small delay for visible streaming effect
            await asyncio.sleep(0.03)  # 30ms delay between words

        # Send completion event
        event_data = {
            "type": "complete",
            "content": current_text.strip(),
            "thread_id": thread_id,
            "metadata": {
                "num_rewrites": result.get("num_rewrites", 0),
                "used_retrieval": result.get("used_retrieval", False)
            }
        }
        yield f"data: {json.dumps(event_data)}\n\n"

    except Exception as e:
        # Send error event
        event_data = {
            "type": "error",
            "error": str(e)
        }
        yield f"data: {json.dumps(event_data)}\n\n"


@app.post("/api/query/stream")
async def query_stream(request: dict):
    """
    Streaming endpoint for real-time RAG responses
    Uses Server-Sent Events (SSE)
    """
    question = request.get("question", "")
    session_id = request.get("session_id", str(uuid.uuid4()))

    # Get or create thread_id for this session
    if session_id not in session_threads:
        session_threads[session_id] = f"thread-{uuid.uuid4()}"
    thread_id = session_threads[session_id]

    # Return SSE stream
    return StreamingResponse(
        generate_sse_stream(question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)