from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from helpers.config import get_settings
from routes.schema import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    SourceChunk,
    CollectionInfoResponse,
)

router = APIRouter()
settings = get_settings()


@router.get("/", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Health-check endpoint — reports API, database, and vector store status."""
    db_connected = False
    vector_db_ready = False

    try:
        await request.app.state.db.command("ping")
        db_connected = True
    except Exception:
        pass

    try:
        info = request.app.state.rag_controller.vector_store.collection_info("lexbridge")
        vector_db_ready = info.get("exists", False)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        db_connected=db_connected,
        vector_db_ready=vector_db_ready,
    )


@router.post("/api/v1/data/upload", response_model=UploadResponse, tags=["Data"])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    collection: str = Form(default="lexbridge"),
):
    """
    Upload and index a document (PDF, DOCX, or TXT).

    The document is parsed, chunked, embedded, and stored in the vector DB.
    Arabic documents are auto-detected and normalized before embedding.
    """
    # Validate content type
    if file.content_type not in settings.FILE_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not allowed. Allowed: {settings.FILE_ALLOWED_EXTENSIONS}",
        )

    # Read and validate size
    content = await file.read()
    max_bytes = settings.FILE_MAX_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.FILE_MAX_SIZE_MB} MB",
        )

    chunks_created = await request.app.state.data_controller.process_file(
        file_content=content,
        filename=file.filename,
        collection=collection,
        chunk_size=settings.FILE_CHUNK_SIZE,
    )

    return UploadResponse(
        message="File processed successfully",
        chunks_created=chunks_created,
        collection=collection,
    )


@router.post("/api/v1/nlp/ask", response_model=QueryResponse, tags=["RAG"])
async def ask_question(request: Request, body: QueryRequest):
    """
    Query the RAG system.

    Accepts a natural-language question, retrieves the top-k relevant chunks
    from the vector store, injects them into the LLM context, and returns
    the generated answer along with source metadata.
    """
    result = await request.app.state.rag_controller.query(
        question=body.text,
        collection=body.collection,
        k=body.k,
        language=body.language.value,
    )

    sources = [SourceChunk(**s) for s in result["sources"]]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        collection=result["collection"],
        model=result["model"],
    )


@router.get(
    "/api/v1/data/collection/{name}",
    response_model=CollectionInfoResponse,
    tags=["Data"],
)
async def collection_info(request: Request, name: str):
    """Return metadata about a vector store collection."""
    try:
        info = request.app.state.rag_controller.vector_store.collection_info(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CollectionInfoResponse(
        collection=name,
        exists=info.get("exists", False),
        points_count=info.get("points_count", 0),
        dimension=info.get("dimension"),
    )
