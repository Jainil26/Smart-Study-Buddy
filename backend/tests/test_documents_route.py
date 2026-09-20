import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.routes.documents import UPLOAD_DIR

MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
    b"2 0 obj <</Type /Pages /Count 1 /Kids [3 0 R]>> endobj\n"
    b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
    b"4 0 obj <</Length 44>> stream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello Smart Study Buddy Chunking Test) Tj ET\n"
    b"endstream endobj\n"
    b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000062 00000 n \n"
    b"0000000130 00000 n \n"
    b"0000000252 00000 n \n"
    b"0000000346 00000 n \n"
    b"trailer <</Size 6 /Root 1 0 R>>\n"
    b"startxref\n"
    b"421\n"
    b"%%EOF\n"
)


class TestDocumentsRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        self.test_filename = "test_chunk_document.pdf"
        self.test_file_path = UPLOAD_DIR / self.test_filename
        with open(self.test_file_path, "wb") as f:
            f.write(MINIMAL_PDF_BYTES)

    def tearDown(self):
        if self.test_file_path.exists():
            self.test_file_path.unlink()

    def test_get_chunks_valid_pdf_end_to_end(self):
        """Verify that an existing PDF returns chunks with correct HTTP 200 response structure."""
        response = self.client.get(f"/api/documents/{self.test_filename}/chunks")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("filename"), self.test_filename)
        self.assertIsInstance(data.get("chunk_count"), int)
        self.assertGreater(data.get("chunk_count"), 0)
        self.assertIsInstance(data.get("chunks"), list)

        first_chunk = data["chunks"][0]
        self.assertEqual(first_chunk.get("chunk_id"), 1)
        self.assertEqual(first_chunk.get("page_number"), 1)
        self.assertIn("Hello Smart Study Buddy", first_chunk.get("text", ""))

    def test_get_chunks_missing_pdf(self):
        """Verify that requesting chunks for a non-existent PDF returns HTTP 404."""
        response = self.client.get("/api/documents/non_existent_file_9999.pdf/chunks")
        self.assertEqual(response.status_code, 404)
        
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    def test_get_chunks_mocked_service_integration(self):
        """Verify endpoint data shape with multiple pages and chunks."""
        mock_pages = {
            "page_count": 2,
            "total_character_count": 1200,
            "total_word_count": 200,
            "pages": [
                {"page_number": 1, "text": "Page one text content. " * 25},
                {"page_number": 2, "text": "Page two text content. " * 25},
            ],
        }

        with patch("app.routes.documents.extract_text_from_pdf", return_value=mock_pages):
            response = self.client.get(f"/api/documents/{self.test_filename}/chunks")
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["filename"], self.test_filename)
            self.assertEqual(data["chunk_count"], len(data["chunks"]))
            self.assertGreater(data["chunk_count"], 1)

            # Verify chunk items match specification
            for idx, chunk in enumerate(data["chunks"], start=1):
                self.assertEqual(chunk["chunk_id"], idx)
                self.assertIn("page_number", chunk)
                self.assertIn("text", chunk)


if __name__ == "__main__":
    unittest.main()
