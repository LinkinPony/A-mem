from typing import List, Dict, Any, Optional
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging

# Assuming BaseRetriever is in memory.storage.base_retriever
from memory.storage.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class ChromaRetriever(BaseRetriever):
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2", db_path: str = "./chroma_db"):
        logger.info(f"Initializing ChromaRetriever with collection: {collection_name}, model: {model_name}, db_path: {db_path}")
        client_settings = Settings(allow_reset=True)
        try:
            self.client = chromadb.PersistentClient(path=db_path, settings=client_settings)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client at path {db_path}: {e}", exc_info=True)
            # Fallback to in-memory if persistent client fails, or re-raise
            logger.warning("Falling back to in-memory ChromaDB client due to error with persistent client.")
            self.client = chromadb.Client(settings=client_settings)


        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Successfully got or created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to get or create collection {collection_name}: {e}", exc_info=True)
            raise # Re-raise the exception if collection setup fails

    def _serialize_metadata(self, metadata: Dict) -> Dict[str, str]:
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            elif value is None:
                processed_metadata[key] = "null" # Or skip, depending on ChromaDB's handling of None
            else:
                processed_metadata[key] = str(value)
        return processed_metadata

    def _deserialize_metadata(self, metadata_str_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if metadata_str_dict is None:
            return {}
        processed_metadata = {}
        for key, value in metadata_str_dict.items():
            try:
                if value == "null":
                    processed_metadata[key] = None
                elif isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    processed_metadata[key] = json.loads(value)
                # Attempt to convert numeric strings back to numbers
                elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        processed_metadata[key] = float(value)
                    else:
                        processed_metadata[key] = int(value)
                else:
                    processed_metadata[key] = value
            except (json.JSONDecodeError, ValueError):
                processed_metadata[key] = value # Keep as string if parsing fails
        return processed_metadata

    def add(self, document: str, metadata: Dict, doc_id: str) -> None:
        logger.debug(f"Adding document id: {doc_id}, metadata: {metadata}")
        serial_metadata = self._serialize_metadata(metadata)
        try:
            self.collection.add(
                documents=[document],
                metadatas=[serial_metadata],
                ids=[doc_id]
            )
            logger.info(f"Document {doc_id} added successfully.")
        except Exception as e:
            logger.error(f"Error adding document {doc_id} to Chroma: {e}", exc_info=True)
            # Decide on error handling: re-raise, or return status, etc.

    def delete(self, doc_id: str) -> None:
        logger.debug(f"Deleting document id: {doc_id}")
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Document {doc_id} deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from Chroma: {e}", exc_info=True)

    def retrieve_by_id(self, doc_id: str) -> Optional[Dict]:
        logger.debug(f"Retrieving document by id: {doc_id}")
        try:
            result = self.collection.get(ids=[doc_id], include=['metadatas', 'documents'])
            if result and result['ids'] and result['ids'][0] == doc_id:
                doc = result['documents'][0] if result['documents'] else None
                meta = self._deserialize_metadata(result['metadatas'][0]) if result['metadatas'] else {}
                if doc is not None:
                    return {"id": doc_id, "document": doc, "metadata": meta}
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id} by ID: {e}", exc_info=True)
            return None

    def retrieve_by_vector(self, vector: List[float], k: int = 5) -> List[Dict]:
        logger.debug(f"Retrieving by vector, k={k}")
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=k,
                include=['metadatas', 'documents', 'distances']
            )
            output = []
            if results and results.get('ids') and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc = results['documents'][0][i] if results['documents'] and results['documents'][0] else None
                    meta = self._deserialize_metadata(results['metadatas'][0][i]) if results['metadatas'] and results['metadatas'][0] else {}
                    dist = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                    if doc is not None:
                         output.append({"id": doc_id, "document": doc, "metadata": meta, "score": dist}) # Chroma uses 'distance'
            return output
        except Exception as e:
            logger.error(f"Error in retrieve_by_vector: {e}", exc_info=True)
            return []

    def retrieve_by_metadata(self, metadata_filter: Dict, k: int = 5, query_texts: Optional[List[str]] = None) -> List[Dict]:
        logger.debug(f"Retrieving by metadata: {metadata_filter}, k={k}")
        # ChromaDB's where filter syntax: e.g., {"source": "my_source"}
        # For complex queries (e.g. $in, $ne), the filter dict needs to be structured accordingly.
        # This implementation assumes direct equality checks for simplicity.
        # Example: metadata_filter = {"tags": {"$contains": "AI"}}
        try:
            query_params = {
                "where": metadata_filter,
                "n_results": k,
                "include": ['metadatas', 'documents', 'distances']
            }
            if query_texts:
                query_params["query_texts"] = query_texts

            results = self.collection.query(**query_params)

            output = []
            if results and results.get('ids') and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc = results['documents'][0][i] if results['documents'] and results['documents'][0] else None
                    meta = self._deserialize_metadata(results['metadatas'][0][i]) if results['metadatas'] and results['metadatas'][0] else {}
                    dist = results['distances'][0][i] if results['distances'] and results['distances'][0] else None # May not always be present/relevant for metadata search without query_texts
                    if doc is not None:
                        entry = {"id": doc_id, "document": doc, "metadata": meta}
                        if dist is not None:
                            entry["score"] = dist
                        output.append(entry)
            return output
        except Exception as e:
            logger.error(f"Error in retrieve_by_metadata: {e}", exc_info=True)
            return []

    def search(self, query: str, k: int = 5) -> Dict:
        logger.debug(f"Performing search with query: '{query}', k={k}")
        # This method should align with the BaseRetriever's expected return type for search
        # which is a dict: {'ids', 'documents', 'metadatas', 'distances'}
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=['metadatas', 'documents', 'distances'] # Ensure all parts are included
            )

            # Deserialize metadata for the output
            if results and results.get('metadatas') and results['metadatas'][0]:
                deserialized_metadatas = [self._deserialize_metadata(meta_dict) for meta_dict in results['metadatas'][0]]
                results['metadatas'][0] = deserialized_metadatas # Replace with deserialized

            # Ensure all keys are present even if empty, to match expected dict structure
            final_results = {
                'ids': results.get('ids', [[]])[0], # Chroma returns list of lists
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'distances': results.get('distances', [[]])[0]
            }
            return final_results
        except Exception as e:
            logger.error(f"Error in search: {e}", exc_info=True)
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}


    def shutdown(self) -> None:
        logger.info("Shutting down ChromaRetriever.")
        try:
            # For PersistentClient, 'reset!' is more of a destructive action.
            # Normal shutdown might involve ensuring client connections are closed if any,
            # but chromadb client typically doesn't require explicit close for PersistentClient.
            # If using an in-memory client for tests, reset might be useful.
            # self.client.reset() # Use with caution, clears the DB.
            # For now, let's assume no specific shutdown action is needed beyond Python's GC
            # unless specific resources need to be released.
            # If 'reset' is intended for cleanup like in tests:
            if hasattr(self.client, 'reset'):
                 self.client.reset()
                 logger.info("ChromaDB client reset.")
            pass
        except Exception as e:
            logger.error(f"Error during ChromaRetriever shutdown: {e}", exc_info=True)