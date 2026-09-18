from typing import Any, Dict, List, Union


def chunk_document(
    extraction_result: Union[Dict[str, Any], List[Dict[str, Any]]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunks page-by-page PDF extraction results into smaller text chunks.

    Args:
        extraction_result: Either the dictionary output from pdf_service.extract_text_from_pdf()
                          (containing a 'pages' list) or a list of page dicts.
        chunk_size: Maximum character count per chunk (default: 500). Must be > 0.
        overlap: Character overlap between consecutive chunks (default: 50). Must be >= 0 and < chunk_size.

    Returns:
        List of dictionaries representing chunks:
            - chunk_id: Sequential integer identifier starting at 1
            - page_number: Source page number (1-based)
            - text: Extracted character chunk
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    # Accept both the full extraction dict and direct list of pages
    if isinstance(extraction_result, dict):
        pages = extraction_result.get("pages", [])
    elif isinstance(extraction_result, list):
        pages = extraction_result
    else:
        raise ValueError("extraction_result must be a dictionary or a list of page objects.")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    step = chunk_size - overlap

    for page in pages:
        page_number = page.get("page_number", 1)
        text = page.get("text", "") or ""

        if not text.strip():
            continue

        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_counter,
                "page_number": page_number,
                "text": chunk_text,
            })
            chunk_counter += 1

            if end == text_len:
                break

            start += step

    return chunks
