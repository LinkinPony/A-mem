from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class BaseRetriever(ABC):
    @abstractmethod
    def add(self, document: str, metadata: Dict, doc_id: str) -> None:
        """Adds a document with its metadata and ID to the storage."""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Deletes a document from the storage by its ID."""
        pass

    @abstractmethod
    def retrieve_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieves a document and its metadata by its ID.
        Returns the document and metadata if found, else None."""
        pass

    @abstractmethod
    def retrieve_by_vector(self, vector: List[float], k: int = 5) -> List[Dict]:
        """Retrieves k documents most similar to the given vector.
        Returns a list of dictionaries, each containing document, metadata, and score."""
        pass

    @abstractmethod
    def retrieve_by_metadata(self, metadata_filter: Dict, k: int = 5, query_texts: Optional[List[str]] = None) -> List[Dict]:
        """Retrieves k documents that match the metadata filter.
        If query_texts are provided, results are further refined by semantic similarity.
        Returns a list of dictionaries, each containing document, metadata, and possibly score."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> Dict:
        """Performs a general search based on a query string.
        This is often a wrapper around retrieve_by_vector or a hybrid search.
        Returns a dictionary containing 'ids', 'documents', 'metadatas', 'distances'."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Performs any necessary cleanup or shutdown operations for the retriever."""
        pass
