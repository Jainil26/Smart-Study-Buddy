import re
from pathlib import Path
from typing import Any, Dict, List, Union
import pypdf


def normalize_whitespace(text: str) -> str:
    """
    Normalizes basic whitespace without destroying paragraph or line structure.
    
    - Standardizes line breaks to \n.
    - Cleans duplicate inline spaces/tabs while keeping line separation intact.
    - Collapses excessive blank lines (3+) down to double newlines (paragraph break).
    """
    if not text:
        return ""

    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean multiple spaces/tabs per line and strip trailing whitespace
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Rejoin lines and collapse multi-blank lines into standard paragraph breaks
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def extract_text_from_pdf(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a PDF file using pypdf.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        Dict containing:
            - page_count: Total number of pages
            - total_character_count: Total character count across all pages
            - total_word_count: Total word count across all pages
            - pages: List of dicts, each with "page_number" (1-based) and "text"
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at path: {path}")

    reader = pypdf.PdfReader(str(path))
    
    extracted_pages: List[Dict[str, Any]] = []
    total_character_count = 0
    total_word_count = 0

    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = normalize_whitespace(raw_text)

        char_count = len(clean_text)
        word_count = len(clean_text.split()) if clean_text else 0

        total_character_count += char_count
        total_word_count += word_count

        extracted_pages.append({
            "page_number": index,
            "text": clean_text
        })

    return {
        "page_count": len(reader.pages),
        "total_character_count": total_character_count,
        "total_word_count": total_word_count,
        "pages": extracted_pages
    }
