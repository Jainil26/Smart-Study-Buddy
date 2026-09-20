import unittest
from app.services.vector_store import VectorStore, cosine_similarity


class TestVectorStore(unittest.TestCase):

    def test_cosine_similarity(self):
        """Verify cosine similarity calculation logic."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        vec_c = [0.0, 1.0, 0.0]
        vec_d = [-1.0, 0.0, 0.0]

        # Identical vectors -> similarity 1.0
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_b), 1.0)

        # Orthogonal vectors -> similarity 0.0
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_c), 0.0)

        # Opposite vectors -> similarity -1.0
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_d), -1.0)

        # Zero vector handling
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([], [1.0]), 0.0)

    def test_store_and_search_descending_order(self):
        """Verify vector store storage and top-k search descending score ordering."""
        store = VectorStore()

        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "TCP is a transport layer protocol."},
            {"chunk_id": 2, "page_number": 1, "text": "UDP is a connectionless transport protocol."},
            {"chunk_id": 3, "page_number": 2, "text": "HTTP is an application layer protocol."}
        ]

        embeddings = [
            [1.0, 0.0, 0.0],   # Chunk 1 vector
            [0.8, 0.6, 0.0],   # Chunk 2 vector
            [0.0, 0.0, 1.0]    # Chunk 3 vector
        ]

        store.add_chunks(chunks, embeddings)
        self.assertEqual(len(store.get_all_chunks()), 3)

        # Query vector close to Chunk 1
        query_vector = [1.0, 0.0, 0.0]

        results = store.search(query_vector, top_k=3)
        self.assertEqual(len(results), 3)

        # First result should be chunk 1 with score 1.0
        self.assertEqual(results[0]["chunk_id"], 1)
        self.assertAlmostEqual(results[0]["score"], 1.0)

        # Second result should be chunk 2 (score 0.8)
        self.assertEqual(results[1]["chunk_id"], 2)
        self.assertAlmostEqual(results[1]["score"], 0.8)

        # Third result should be chunk 3 (score 0.0)
        self.assertEqual(results[2]["chunk_id"], 3)
        self.assertAlmostEqual(results[2]["score"], 0.0)

    def test_top_k_truncation(self):
        """Verify top_k parameter limits the number of returned chunks."""
        store = VectorStore()
        chunks = [
            {"chunk_id": i, "page_number": 1, "text": f"Chunk {i}"} for i in range(1, 6)
        ]
        embeddings = [[float(i), 1.0] for i in range(1, 6)]

        store.add_chunks(chunks, embeddings)

        results = store.search([5.0, 1.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], 5)

    def test_search_preserves_chunk_id(self):
        """Verify that chunk_id is present in every result returned by search()."""
        store = VectorStore()
        store.add_chunk(chunk_id=42, page_number=3, text="Sample text", embedding=[1.0, 0.0])

        results = store.search([1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertIn("chunk_id", result)
        self.assertEqual(result["chunk_id"], 42)
        self.assertIn("page_number", result)
        self.assertEqual(result["page_number"], 3)
        self.assertIn("score", result)
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Sample text")


if __name__ == "__main__":
    unittest.main()

