import unittest
from app.services.chunk_service import chunk_document


class TestChunkService(unittest.TestCase):

    def setUp(self):
        self.sample_extraction_result = {
            "page_count": 2,
            "total_character_count": 1200,
            "total_word_count": 200,
            "pages": [
                {
                    "page_number": 1,
                    "text": "A" * 1000  # Page 1: 1000 'A' characters
                },
                {
                    "page_number": 2,
                    "text": "B" * 300   # Page 2: 300 'B' characters
                }
            ]
        }

    def test_basic_chunking(self):
        chunks = chunk_document(self.sample_extraction_result, chunk_size=500, overlap=50)
        
        # Page 1 (1000 chars): chunk 1 [0..500], chunk 2 [450..950], chunk 3 [900..1000] -> 3 chunks
        # Page 2 (300 chars): chunk 4 [0..300] -> 1 chunk
        # Total chunks = 4
        self.assertEqual(len(chunks), 4)

        # Check fields present in every chunk
        for idx, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["chunk_id"], idx)
            self.assertIn("page_number", chunk)
            self.assertIn("text", chunk)

    def test_page_number_preservation(self):
        chunks = chunk_document(self.sample_extraction_result, chunk_size=500, overlap=50)
        
        # Chunks 1, 2, 3 should belong to page 1
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[1]["page_number"], 1)
        self.assertEqual(chunks[2]["page_number"], 1)

        # Chunk 4 should belong to page 2
        self.assertEqual(chunks[3]["page_number"], 2)

    def test_overlap_preservation(self):
        page_text = "0123456789" * 10  # 100 characters
        extraction = {"pages": [{"page_number": 1, "text": page_text}]}
        
        # chunk_size=40, overlap=10 -> step=30
        # Chunk 1: [0..40]
        # Chunk 2: [30..70]
        # Chunk 3: [60..100]
        chunks = chunk_document(extraction, chunk_size=40, overlap=10)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["text"], page_text[0:40])
        self.assertEqual(chunks[1]["text"], page_text[30:70])
        self.assertEqual(chunks[2]["text"], page_text[60:100])

        # Verify overlap between chunk 0 and chunk 1 is 10 chars
        self.assertEqual(chunks[0]["text"][-10:], chunks[1]["text"][:10])

    def test_list_input(self):
        pages_list = [
            {"page_number": 1, "text": "Page one text content."},
            {"page_number": 2, "text": "Page two text content."}
        ]
        chunks = chunk_document(pages_list, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[1]["page_number"], 2)

    def test_empty_pages_handled(self):
        extraction = {
            "pages": [
                {"page_number": 1, "text": "  "},
                {"page_number": 2, "text": "Valid content on page two."}
            ]
        }
        chunks = chunk_document(extraction, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_number"], 2)
        self.assertEqual(chunks[0]["text"], "Valid content on page two.")

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_document(self.sample_extraction_result, chunk_size=0, overlap=10)

        with self.assertRaises(ValueError):
            chunk_document(self.sample_extraction_result, chunk_size=100, overlap=-5)

        with self.assertRaises(ValueError):
            chunk_document(self.sample_extraction_result, chunk_size=100, overlap=100)

        with self.assertRaises(ValueError):
            chunk_document("invalid_input")


if __name__ == "__main__":
    unittest.main()
