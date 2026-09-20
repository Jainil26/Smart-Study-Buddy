import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.chunk_service import chunk_document
from app.services.pdf_service import extract_text_from_pdf
from app.services.retrieval_service import search_document

router = APIRouter(prefix="/api/documents", tags=["Documents"])

# Path to the uploads directory: backend/uploads
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a single PDF file and save it to backend/uploads/.
    """
    filename = file.filename or ""

    # Verify that the uploaded file is a PDF (check content-type and file extension)
    is_pdf_type = file.content_type == "application/pdf"
    is_pdf_ext = filename.lower().endswith(".pdf")

    if not (is_pdf_type or is_pdf_ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )

    # Ensure the uploads directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Safely get the base filename
    safe_filename = Path(filename).name

    # Destination path for the saved file
    file_path = UPLOAD_DIR / safe_filename

    # Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "filename": filename,
        "saved_filename": safe_filename
    }


@router.get("/{filename}/extract")
async def extract_pdf_text(filename: str):
    """
    Endpoint to extract text page-by-page from an existing PDF file in backend/uploads/.
    """
    safe_filename = Path(filename).name
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in uploads directory."
        )

    try:
        extraction_result = extract_text_from_pdf(file_path)
        return {
            "success": True,
            **extraction_result
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract text from PDF."
        )


@router.get("/{filename}/chunks")
async def get_document_chunks(filename: str):
    """
    Endpoint to inspect generated chunks for an uploaded PDF file in backend/uploads/.
    """
    safe_filename = Path(filename).name
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in uploads directory."
        )

    try:
        extraction_result = extract_text_from_pdf(file_path)
        chunks = chunk_document(extraction_result, chunk_size=500, overlap=50)
        return {
            "success": True,
            "filename": filename,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate document chunks."
        )


@router.get("/{filename}/search")
async def search_document_chunks_route(filename: str, q: str = "", top_k: int = 3):
    """
    Endpoint to search document chunks using cosine similarity on embeddings.
    """
    safe_filename = Path(filename).name
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in uploads directory."
        )

    if not q or not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' cannot be empty."
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'top_k' must be a positive integer."
        )

    try:
        results = search_document(file_path=file_path, query=q.strip(), top_k=top_k)
        return {
            "success": True,
            "filename": filename,
            "query": q.strip(),
            "result_count": len(results),
            "results": results
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search document chunks: {exc}"
        )


