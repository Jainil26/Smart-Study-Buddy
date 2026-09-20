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
    b"BT /F1 12 Tf 100 700 Td (TCP is Transmission Control Protocol) Tj ET\n"
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


class TestSearchRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        self.test_filename = "test_search_doc.pdf"
        self.test_file_path = UPLOAD_DIR / self.test_filename
        with open(self.test_file_path, "wb") as f:
            f.write(MINIMAL_PDF_BYTES)

    def tearDown(self):
        if self.test_file_path.exists():
            self.test_file_path.unlink()

    @patch("app.services.retrieval_service.generate_embedding")
    @patch("app.services.retrieval_service.generate_embeddings")
    def test_search_valid_query(self, mock_gen_embeddings, mock_gen_embedding):
        """Verify successful search endpoint response."""
        mock_gen_embeddings.return_value = [[1.0, 0.0, 0.0]]
        mock_gen_embedding.return_value = [1.0, 0.0, 0.0]

        response = self.client.get(
            f"/api/documents/{self.test_filename}/search?q=What%20is%20TCP?&top_k=3"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("filename"), self.test_filename)
        self.assertEqual(data.get("query"), "What is TCP?")
        self.assertEqual(data.get("result_count"), 1)

        results = data.get("results")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        first_res = results[0]
        self.assertEqual(first_res.get("chunk_id"), 1)
        self.assertEqual(first_res.get("page_number"), 1)
        self.assertIn("TCP is Transmission Control Protocol", first_res.get("text"))
        self.assertIn("score", first_res)

    def test_search_missing_pdf_returns_404(self):
        """Verify requesting search on a non-existent PDF returns HTTP 404."""
        response = self.client.get("/api/documents/non_existent_file.pdf/search?q=Test")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    def test_search_empty_query_returns_400(self):
        """Verify empty or whitespace-only query parameter 'q' returns HTTP 400."""
        response = self.client.get(f"/api/documents/{self.test_filename}/search?q=")
        self.assertEqual(response.status_code, 400)

        response_space = self.client.get(f"/api/documents/{self.test_filename}/search?q=%20%20")
        self.assertEqual(response_space.status_code, 400)

    def test_search_invalid_top_k_returns_400(self):
        """Verify non-positive top_k query parameter returns HTTP 400."""
        response_zero = self.client.get(f"/api/documents/{self.test_filename}/search?q=TCP&top_k=0")
        self.assertEqual(response_zero.status_code, 400)

        response_neg = self.client.get(f"/api/documents/{self.test_filename}/search?q=TCP&top_k=-5")
        self.assertEqual(response_neg.status_code, 400)


if __name__ == "__main__":
    unittest.main()
