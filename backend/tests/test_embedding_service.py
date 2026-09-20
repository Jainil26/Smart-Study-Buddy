import os
import unittest
from unittest.mock import MagicMock, patch

from app.services.embedding_service import (
    generate_embedding,
    generate_embeddings,
    get_hf_token,
    get_model_name,
)


class TestEmbeddingService(unittest.TestCase):

    def test_missing_hf_token_raises_value_error(self):
        """Verify that a missing or empty HF_TOKEN raises a ValueError."""
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_hf_token()
            self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_model_name_default_and_custom(self):
        """Verify that HF_EMBEDDING_MODEL environment variable works correctly."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_model_name(), "sentence-transformers/all-MiniLM-L6-v2")

        with patch.dict(os.environ, {"HF_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"}):
            self.assertEqual(get_model_name(), "BAAI/bge-small-en-v1.5")

    @patch("app.services.embedding_service.requests.post")
    def test_generate_embedding_single(self, mock_post):
        """Verify generating a single text embedding with mocked HF API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"HF_TOKEN": "mock_hf_token_123"}):
            vec = generate_embedding("What is TCP?")

        self.assertEqual(vec, [0.1, 0.2, 0.3, 0.4])
        mock_post.assert_called_once()
        headers = mock_post.call_args.kwargs.get("headers", {})
        self.assertEqual(headers.get("Authorization"), "Bearer mock_hf_token_123")

    @patch("app.services.embedding_service.requests.post")
    def test_generate_embeddings_batch(self, mock_post):
        """Verify generating multiple text embeddings in batch with mocked HF API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"HF_TOKEN": "mock_hf_token_123"}):
            vectors = generate_embeddings(["Text 1", "Text 2"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], [0.1, 0.2, 0.3])
        self.assertEqual(vectors[1], [0.4, 0.5, 0.6])

    def test_generate_embedding_empty_text_raises_error(self):
        """Verify that empty or whitespace-only text raises a ValueError."""
        with patch.dict(os.environ, {"HF_TOKEN": "mock_hf_token_123"}):
            with self.assertRaises(ValueError):
                generate_embedding("   ")


if __name__ == "__main__":
    unittest.main()
