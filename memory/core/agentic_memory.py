import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from llm_controller import LLMController
# Assuming BaseRetriever is in memory.storage.base_retriever
from memory.storage.base_retriever import BaseRetriever
from memory.note import MemoryNote
# Assuming Analyzer is now in memory.core.analyzer
from memory.core.analyzer import Analyzer
# Import Evolver's decision models and _extract_json_from_response from utils
from memory.utils import _extract_json_from_response
# The Pydantic models from Evolver will be used directly or redefined here
from pydantic import BaseModel, Field
from typing import Union # For Union type hint

logger = logging.getLogger(__name__)

# Helper function from old AgenticMemorySystem, may be useful
def _note_to_metadata_for_retriever(note: MemoryNote) -> Dict:
    """Converts MemoryNote object to a dictionary suitable for retriever metadata.
       Ensures all values are serializable (e.g., lists/dicts to JSON strings if necessary,
       though ChromaRetriever's _serialize_metadata handles this)."""
    return {
        "id": note.id,
        "content": note.content, # Content is stored as document, but can be in metadata too
        "keywords": note.keywords,
        "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "context": note.context,
        "evolution_history": note.evolution_history,
        "category": note.category,
        "tags": note.tags
    }

# Pydantic models from Evolver (StrengthenAction, UpdateNeighborAction, EvolutionDecision)
# These are needed for the evolution logic embedded in AgenticMemory.
class StrengthenAction(BaseModel):
    action_type: str = Field(default="strengthen", pattern="^strengthen$")
    target_memory_id: str = Field(..., description="ID of the neighboring memory to strengthen connection with.")
    new_tags_for_current_note: List[str] = Field(..., description="Updated list of tags for the current memory.")

class UpdateNeighborAction(BaseModel):
    action_type: str = Field(default="update_neighbor", pattern="^update_neighbor$")
    target_memory_id: str = Field(..., description="ID of the neighboring memory to update.")
    new_context: str = Field(..., description="New context for the target neighboring memory.")
    new_tags: List[str] = Field(..., description="New list of tags for the target neighboring memory.")

class EvolutionDecision(BaseModel):
    should_evolve: bool = Field(..., description="Whether the current memory should undergo evolution.")
    actions: List[Union[StrengthenAction, UpdateNeighborAction]] = Field(default=[], description="List of evolution actions to perform.")

_EVOLUTION_SYSTEM_PROMPT = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Your task is to analyze a new memory note in the context of its nearest neighbors from the knowledge base and decide how to evolve the memories.

**New Memory Note Details:**
- Content: {content}
- Context: {context}
- Keywords: {keywords}
- ID: {note_id}

**Nearest Neighbors in Knowledge Base:**
{nearest_neighbors_memories}

**Your Decision-Making Process:**
1.  **Analyze Relationships:** Compare the new note with its neighbors.
2.  **Decide to Evolve:** Based on the analysis, decide if any evolution is needed. Set `should_evolve` to `true` if so.
3.  **Define Actions:** If `should_evolve` is true, create a list of actions. Each action is a JSON object.
    - To **strengthen**, this links the new note to ONE of its neighbors. The object MUST contain `"action_type": "strengthen"`. For the `target_memory_id`, you MUST choose the ID from one of the neighbors listed above in the "Nearest Neighbors in Knowledge Base" section. It also requires `new_tags_for_current_note` for the new note itself.
    - To **update a neighbor**, the object MUST contain `"action_type": "update_neighbor"`. It also requires `target_memory_id`, `new_context`, and `new_tags`.

Return your decision in the required JSON format.
'''

class AgenticMemory:
    def __init__(self, retriever: BaseRetriever, llm_controller: LLMController):
        self.retriever = retriever
        self.llm_controller = llm_controller
        self.analyzer = Analyzer(llm_controller=self.llm_controller)
        self.memories: Dict[str, MemoryNote] = {} # In-memory cache/store
        # Load existing memories from retriever? For now, we assume it starts empty or
        # that QueryPlanner will be responsible for populating it if needed for complex queries.
        # Or, AgenticMemory could have a load_all() method.
        # For now, focus on the core add/get/update/delete logic.

    def _find_related_memories_for_evolution(self, query_content: str, k: int = 5) -> Tuple[str, List[str]]:
        """Finds related memories using the retriever and formats them for the Evolver LLM prompt."""
        if not self.memories: # Or if retriever has no data
            return "", []

        try:
            # Use the retriever's search or retrieve_by_vector method
            # The old system used retriever.search() which returned a dict
            # The Evolver expected a text blob and a list of IDs.
            search_results = self.retriever.search(query=query_content, k=k)

            neighbor_ids = search_results.get('ids', [])
            if not neighbor_ids:
                return "", []

            memory_parts = []
            valid_neighbor_ids = []
            for i, doc_id in enumerate(neighbor_ids):
                # Attempt to get the full note from in-memory cache first for consistency
                note = self.memories.get(doc_id)
                if note:
                    memory_parts.append(
                        f"ID: {note.id}\n"
                        f"Content: {note.content}\n"
                        f"Context: {note.context}\n"
                        f"Tags: {note.tags}"
                    )
                    valid_neighbor_ids.append(note.id)
                elif search_results.get('documents') and search_results.get('metadatas'):
                    # Fallback to retriever results if not in local cache (e.g. just added)
                    # This path might be less common if all active notes are in self.memories
                    doc_content = search_results['documents'][i]
                    meta = search_results['metadatas'][i]
                    memory_parts.append(
                        f"ID: {doc_id}\n"
                        f"Content: {doc_content}\n"
                        f"Context: {meta.get('context', 'N/A')}\n"
                        f"Tags: {meta.get('tags', [])}"
                    )
                    valid_neighbor_ids.append(doc_id)

            memory_str = "\n---\n".join(memory_parts)
            return memory_str, valid_neighbor_ids
        except Exception as e:
            logger.error(f"Error in _find_related_memories_for_evolution: {str(e)}", exc_info=True)
            return "", []

    def add(self, content: str, time: Optional[str] = None, **kwargs) -> str:
        analysis_results = self.analyzer.analyze_content(content)

        note_kwargs = {
            'keywords': kwargs.get('keywords', analysis_results.get('keywords', [])),
            'context': kwargs.get('context', analysis_results.get('context', "General")),
            'tags': kwargs.get('tags', analysis_results.get('tags', [])),
            'timestamp': time if time is not None else datetime.now().strftime("%Y%m%d%H%M%S%f"),
        }
        # Add any other kwargs passed directly, like category
        for key, value in kwargs.items():
            if key not in note_kwargs and hasattr(MemoryNote, key):
                note_kwargs[key] = value

        note_kwargs = {k: v for k, v in note_kwargs.items() if v is not None}
        note = MemoryNote(content=content, **note_kwargs)
        self.memories[note.id] = note # Add to in-memory dict

        # --- Evolution Logic (adapted from Evolver class) ---
        evolved = False
        try:
            neighbors_text, neighbor_ids = self._find_related_memories_for_evolution(note.content, 5)
            if neighbors_text and neighbor_ids:
                prompt = _EVOLUTION_SYSTEM_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    keywords=str(note.keywords), # Ensure keywords are string for prompt
                    note_id=note.id,
                    nearest_neighbors_memories=neighbors_text,
                )
                response_str = self.llm_controller.get_completion(
                    prompt,
                    response_format={"type": "json_object", "schema": EvolutionDecision.model_json_schema()},
                    stage="Memory Evolution"
                )
                clean_json_str = _extract_json_from_response(response_str)
                if clean_json_str:
                    decision = EvolutionDecision.model_validate_json(clean_json_str)
                    if decision.should_evolve:
                        evolved = True
                        for action in decision.actions:
                            if isinstance(action, StrengthenAction):
                                if action.target_memory_id in neighbor_ids:
                                    note.links.append(action.target_memory_id)
                                    note.tags = action.new_tags_for_current_note # Update current note's tags
                                    logger.info(f"Strengthened note {note.id} with link to {action.target_memory_id}")

                                    # Add backlink to neighbor and persist neighbor
                                    neighbor_note = self.memories.get(action.target_memory_id)
                                    if neighbor_note:
                                        if note.id not in neighbor_note.links:
                                            neighbor_note.links.append(note.id)
                                            self.update(neighbor_note.id, links=neighbor_note.links) # Persists neighbor
                                    # If neighbor not in self.memories, it implies an issue or it's only in retriever
                                    # For now, we assume active processing involves self.memories

                            elif isinstance(action, UpdateNeighborAction):
                                if action.target_memory_id in self.memories: # Check if neighbor is in memory
                                    self.update(
                                        action.target_memory_id,
                                        context=action.new_context,
                                        tags=action.new_tags
                                    ) # Persists neighbor
                                    logger.info(f"Updated neighbor note {action.target_memory_id}")
        except Exception as e:
            logger.error(f"Error during memory evolution for note {note.id}: {e}", exc_info=True)
        # --- End Evolution Logic ---

        # Persist the (potentially evolved) new note
        self.retriever.add(document=note.content, metadata=_note_to_metadata_for_retriever(note), doc_id=note.id)
        logger.info(f"Added note {note.id} to retriever. Evolved: {evolved}")
        return note.id

    def get(self, memory_id: str) -> Optional[MemoryNote]:
        note = self.memories.get(memory_id)
        if not note:
            # Try to fetch from retriever if not in memory
            retrieved_data = self.retriever.retrieve_by_id(memory_id)
            if retrieved_data:
                # Reconstruct MemoryNote. This assumes metadata stores all necessary fields.
                # This might be simplified if MemoryNote has a from_dict/from_retriever_data constructor.
                meta = retrieved_data.get("metadata", {})
                content = retrieved_data.get("document")
                if content:
                    # Ensure all required fields for MemoryNote are present in meta or have defaults
                    note_args = {
                        "id": memory_id,
                        "content": content,
                        "keywords": meta.get("keywords", []),
                        "links": meta.get("links", []),
                        "retrieval_count": meta.get("retrieval_count", 0),
                        "timestamp": meta.get("timestamp", datetime.now().strftime("%Y%m%d%H%M%S%f")),
                        "last_accessed": meta.get("last_accessed", datetime.now().strftime("%Y%m%d%H%M%S%f")),
                        "context": meta.get("context", "General"),
                        "evolution_history": meta.get("evolution_history", []),
                        "category": meta.get("category", "Uncategorized"),
                        "tags": meta.get("tags", [])
                    }
                    note = MemoryNote(**note_args)
                    self.memories[memory_id] = note # Cache it

        if note:
            note.retrieval_count += 1
            note.last_accessed = datetime.now().strftime("%Y%m%d%H%M%S%f")
            # Update in retriever if retrieval count/last_accessed update is desired to be persistent
            # self.retriever.add(document=note.content, metadata=_note_to_metadata_for_retriever(note), doc_id=note.id)
            # For now, keep this update in-memory only to avoid write on read, unless explicitly needed.
        return note


    def update(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.memories:
            # Optionally, try to load from retriever first
            if not self.get(memory_id): # This will load it into self.memories if found
                 logger.warning(f"Attempted to update non-existent memory_id: {memory_id}")
                 return False

        note = self.memories[memory_id]
        updated_fields = False
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                updated_fields = True

        if updated_fields:
            note.last_accessed = datetime.now().strftime("%Y%m%d%H%M%S%f")
            # If content is updated, should we re-analyze?
            # Current plan: No, assume metadata is provided or existing is fine.
            # For evolution, if content changes, it might trigger re-evolution indirectly
            # if it's re-added or if update triggers an evolution check.
            # For now, update is direct.

            self.retriever.add(document=note.content, metadata=_note_to_metadata_for_retriever(note), doc_id=note.id)
            logger.info(f"Updated note {memory_id} in retriever.")
            return True
        return False

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.retriever.delete(doc_id=memory_id)
            logger.info(f"Deleted note {memory_id} from memory and retriever.")
            return True
        # If not in self.memories, still try to delete from retriever
        try:
            self.retriever.delete(doc_id=memory_id)
            logger.info(f"Deleted note {memory_id} from retriever (was not in local memory).")
            return True # Return true if deletion from retriever was attempted/successful
        except Exception as e:
            logger.error(f"Error deleting note {memory_id} from retriever: {e}", exc_info=True)
            return False

    def search_notes_by_content(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """ Searches notes by content similarity using the retriever and returns rich dictionary objects.
            This is a lower-level search, results from which QueryPlanner might consume.
        """
        search_results_dict = self.retriever.search(query, k) # Expects dict: ids, documents, metadatas, distances

        memories_found = []
        if search_results_dict and search_results_dict.get('ids'):
            for i, doc_id in enumerate(search_results_dict['ids']):
                # Try to get full MemoryNote object if available for richer data
                note_obj = self.memories.get(doc_id)
                if note_obj:
                    content = note_obj.content
                    context = note_obj.context
                    keywords = note_obj.keywords
                    tags = note_obj.tags
                    timestamp = note_obj.timestamp
                    category = note_obj.category
                else: # Fallback to retriever metadata
                    meta = search_results_dict['metadatas'][i] if search_results_dict.get('metadatas') and len(search_results_dict['metadatas']) > i else {}
                    content = search_results_dict['documents'][i] if search_results_dict.get('documents') and len(search_results_dict['documents']) > i else "N/A"
                    context = meta.get('context', 'N/A')
                    keywords = meta.get('keywords', [])
                    tags = meta.get('tags', [])
                    timestamp = meta.get('timestamp', 'N/A')
                    category = meta.get('category', 'Uncategorized')

                score = search_results_dict['distances'][i] if search_results_dict.get('distances') and len(search_results_dict['distances']) > i else 0.0

                memories_found.append({
                    'id': doc_id,
                    'content': content,
                    'context': context,
                    'keywords': keywords,
                    'tags': tags,
                    'timestamp': timestamp,
                    'category': category,
                    'score': score
                })
        return memories_found[:k]

    def get_all_memory_ids(self) -> List[str]:
        """Returns a list of all memory IDs currently in the in-memory store."""
        return list(self.memories.keys())

    def load_memory_from_retriever(self, memory_id: str) -> Optional[MemoryNote]:
        """Explicitly loads a single memory from the retriever into the in-memory cache."""
        if memory_id in self.memories:
            return self.memories[memory_id]

        retrieved_data = self.retriever.retrieve_by_id(memory_id)
        if retrieved_data:
            meta = retrieved_data.get("metadata", {})
            content = retrieved_data.get("document")
            if content:
                note_args = {
                    "id": memory_id, "content": content,
                    "keywords": meta.get("keywords", []), "links": meta.get("links", []),
                    "retrieval_count": meta.get("retrieval_count", 0),
                    "timestamp": meta.get("timestamp", datetime.now().strftime("%Y%m%d%H%M%S%f")),
                    "last_accessed": meta.get("last_accessed", datetime.now().strftime("%Y%m%d%H%M%S%f")),
                    "context": meta.get("context", "General"),
                    "evolution_history": meta.get("evolution_history", []),
                    "category": meta.get("category", "Uncategorized"),
                    "tags": meta.get("tags", [])
                }
                note = MemoryNote(**note_args)
                self.memories[memory_id] = note
                return note
        return None

    def load_all_memories_from_retriever(self) -> None:
        """ Loads all memories from the retriever into the in-memory store.
        This could be memory intensive for large databases.
        The retriever interface doesn't have a get_all() method, so this would require
        querying with a broad filter or iterating if the DB supports it.
        For Chroma, one might query with no filter and a large k, but this is not ideal.
        Placeholder: This functionality needs a proper way to get all docs from BaseRetriever.
        For now, let's assume this is not called or handled by QueryPlanner if needed.
        A simple way for Chroma would be collection.get() with no IDs.
        """
        logger.warning("load_all_memories_from_retriever is not fully implemented due to BaseRetriever limitations for 'get all'.")
        # Example if retriever supported collection.get() without ids to fetch all:
        # all_data = self.retriever.collection.get(include=['metadatas', 'documents']) # This is Chroma specific
        # for i, doc_id in enumerate(all_data['ids']):
        #    if doc_id not in self.memories:
        #        # Reconstruct and add to self.memories
        #        pass
        pass
