from pathlib import Path
from typing import Any, Dict, List, Union

from app.services.chunk_service import chunk_document
from app.services.embedding_service import generate_embedding, generate_embeddings
from app.services.pdf_service import extract_text_from_pdf
from app.services.vector_store import VectorStore


def search_document(
    file_path: Union[str, Path],
    query: str,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF, chunks it, generates embeddings, and retrieves
    the top-k most relevant chunks matching the query.

    Args:
        file_path: Path to the PDF file.
        query: Natural language query string.
        top_k: Number of relevant chunks to retrieve.
        chunk_size: Maximum character count per chunk.
        overlap: Overlap character count between chunks.

    Returns:
        List of dictionaries containing chunk retrieval results:
            - chunk_id: int
            - page_number: int
            - score: float
            - text: str
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at path: {path}")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty or whitespace-only.")

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # 1. Extract text page-by-page from PDF
    extraction_result = extract_text_from_pdf(path)

    # 2. Chunk document text
    chunks = chunk_document(extraction_result, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return []

    # 3. Generate embeddings for all chunk texts
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = generate_embeddings(chunk_texts)

    # 4. Populate lightweight local vector store
    store = VectorStore()
    store.add_chunks(chunks, chunk_embeddings)

    # 5. Generate embedding for user query
    query_embedding = generate_embedding(query.strip())

    # 6. Search vector store and return top-k chunks
    results = store.search(query_embedding, top_k=top_k)
    return results
