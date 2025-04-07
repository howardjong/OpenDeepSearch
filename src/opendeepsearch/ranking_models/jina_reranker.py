import requests
import torch
from typing import List, Optional
from dotenv import load_dotenv
import os
import warnings
import logging
from .base_reranker import BaseSemanticSearcher

# Configure logging
logger = logging.getLogger(__name__)

class JinaReranker(BaseSemanticSearcher):
    """
    Semantic searcher implementation using Jina AI's embedding API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "jina-embeddings-v3"):
        """
        Initialize the Jina reranker.

        Args:
            api_key: Jina AI API key. If None, will load from environment variable JINA_API_KEY
            model: Model name to use (default: "jina-embeddings-v3")
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('JINA_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and JINA_API_KEY not found in environment variables")

        self.api_url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.logger.info("JinaReranker initialized")

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts using Jina AI API.

        Args:
            texts: List of text strings to embed

        Returns:
            torch.Tensor containing the embeddings
        """
        data = {
            "model": self.model,
            "task": "text-matching",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()  # Raise exception for non-200 status codes

            # Extract embeddings from response
            embeddings_data = [item["embedding"] for item in response.json()["data"]]

            # Convert to torch tensor
            embeddings = torch.tensor(embeddings_data)

            return embeddings

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Jina AI API: {str(e)}")

    def rerank(self, query, documents, max_results=10):
        """
        Rerank documents based on their relevance to the query
        """
        if not documents:
            self.logger.warning("No documents to rerank")
            return []

        endpoint = "https://api.jina.ai/v1/rerank"
        payload = {
            "query": query,
            "documents": documents,
            "top_k": min(max_results, len(documents))
        }

        try:
            self.logger.info(f"Sending rerank request for query: '{query[:50]}...' with {len(documents)} documents")
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=30  # Add timeout to prevent hanging
            )
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Reranking successful, received {len(result.get('results', []))} results")

            # Return the sorted documents
            return [doc["content"] for doc in result["results"]]

        except requests.exceptions.Timeout:
            self.logger.error("Reranking request timed out")
            # Return original documents if timeout occurs
            return documents[:max_results]

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Reranking request failed: {str(e)}")
            # Return original documents if request fails
            return documents[:max_results]

        except Exception as e:
            self.logger.error(f"Unexpected error during reranking: {str(e)}")
            # Return original documents if any other error occurs
            return documents[:max_results]