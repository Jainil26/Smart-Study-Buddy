import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.retrieval_service import search_document

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


class TestRetrievalService(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(__file__).parent / "tmp"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_path = self.test_dir / "test_sample.pdf"
        with open(self.pdf_path, "wb") as f:
            f.write(MINIMAL_PDF_BYTES)

    def tearDown(self):
        if self.pdf_path.exists():
            self.pdf_path.unlink()

    @patch("app.services.retrieval_service.generate_embedding")
    @patch("app.services.retrieval_service.generate_embeddings")
    def test_search_document_end_to_end_mocked(self, mock_gen_embeddings, mock_gen_embedding):
        """Verify search_document workflow with mocked embedding service calls."""
        # Mock chunk embedding generation
        mock_gen_embeddings.return_value = [[1.0, 0.0, 0.0]]
        # Mock query embedding generation
        mock_gen_embedding.return_value = [1.0, 0.0, 0.0]

        results = search_document(self.pdf_path, query="What is TCP?", top_k=3)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        first_res = results[0]
        self.assertEqual(first_res["chunk_id"], 1)
        self.assertEqual(first_res["page_number"], 1)
        self.assertIn("TCP is Transmission Control Protocol", first_res["text"])
        self.assertAlmostEqual(first_res["score"], 1.0)

    def test_search_document_missing_file_raises_not_found(self):
        """Verify searching a non-existent PDF file raises FileNotFoundError."""
        missing_path = self.test_dir / "does_not_exist.pdf"
        with self.assertRaises(FileNotFoundError):
            search_document(missing_path, query="Test Query")

    def test_search_document_invalid_args_raises_value_error(self):
        """Verify invalid query or top_k raises ValueError."""
        with self.assertRaises(ValueError):
            search_document(self.pdf_path, query="   ")

        with self.assertRaises(ValueError):
            search_document(self.pdf_path, query="Valid Query", top_k=0)


if __name__ == "__main__":
    unittest.main()
