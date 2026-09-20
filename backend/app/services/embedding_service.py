import os
from typing import List, Union
import requests

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_hf_token() -> str:
    """
    Retrieves the Hugging Face API token from environment variables.
    Raises ValueError if HF_TOKEN is missing or empty.
    """
    token = os.getenv("HF_TOKEN")
    if not token or not token.strip():
        raise ValueError(
            "HF_TOKEN environment variable is missing or empty. "
            "Please provide a valid Hugging Face API token in environment variables or .env file."
        )
    return token.strip()


def get_model_name() -> str:
    """
    Retrieves the Hugging Face embedding model name from environment variables.
    Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    model = os.getenv("HF_EMBEDDING_MODEL")
    if model and model.strip():
        return model.strip()
    return DEFAULT_MODEL


def _extract_embedding_vector(data: Union[List, float]) -> List[float]:
    """
    Parses a single embedding item returned by HF Inference API.
    Handles 1D embedding list or 2D token-level list (performing mean pooling).
    """
    if not isinstance(data, list):
        raise ValueError(f"Unexpected embedding data format received from API: {type(data)}")

    if not data:
        return []

    # If 1D list of numbers
    if isinstance(data[0], (int, float)):
        return [float(x) for x in data]

    # If 2D list (token embeddings for a single text)
    if isinstance(data[0], list):
        # Check if 2D numerical list (tokens)
        if len(data[0]) > 0 and isinstance(data[0][0], (int, float)):
            dim = len(data[0])
            num_tokens = len(data)
            return [sum(token[i] for token in data) / num_tokens for i in range(dim)]

    raise ValueError(f"Could not parse embedding vector from HF API response: {data}")


def _query_hf_api(payload: dict) -> Union[List, dict]:
    """
    Sends a HTTP POST request to the Hugging Face Inference API.
    Isolates Hugging Face network API code.
    """
    token = get_hf_token()
    model_name = get_model_name()
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Hugging Face API request failed: {exc}") from exc


def generate_embedding(text: str) -> List[float]:
    """
    Generates an embedding vector for a single string of text using Hugging Face Inference API.

    Args:
        text: Input string to embed.

    Returns:
        List[float]: Vector representing text embedding.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    response_data = _query_hf_api(payload)

    # Response can be 1D list of floats [d1, d2, ...]
    # or 2D list of token embeddings [[t1...], [t2...]]
    # or nested list [[d1, d2, ...]]
    if isinstance(response_data, list):
        if len(response_data) > 0 and isinstance(response_data[0], list) and len(response_data) == 1 and isinstance(response_data[0][0], (int, float)):
            return [float(x) for x in response_data[0]]
        return _extract_embedding_vector(response_data)

    raise ValueError(f"Unexpected response structure from Hugging Face API: {response_data}")


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embedding vectors for a list of text strings using Hugging Face Inference API.

    Args:
        texts: List of input strings to embed.

    Returns:
        List[List[float]]: List of embedding vectors.
    """
    if not texts:
        return []

    valid_texts = [t if (t and t.strip()) else " " for t in texts]

    payload = {
        "inputs": valid_texts,
        "options": {"wait_for_model": True}
    }

    response_data = _query_hf_api(payload)

    if not isinstance(response_data, list):
        raise ValueError(f"Unexpected batch response structure from Hugging Face API: {response_data}")

    # If response is a 2D list of embeddings [[d1, d2...], [d1, d2...]]
    embeddings: List[List[float]] = []
    for item in response_data:
        vec = _extract_embedding_vector(item)
        embeddings.append(vec)

    return embeddings
