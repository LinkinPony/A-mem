import logging
from typing import List, Dict, Any, Optional

# Assuming AgenticMemory is in memory.core.agentic_memory
from memory.core.agentic_memory import AgenticMemory
from llm_controller import LLMController # Optional: if QueryPlanner uses LLM directly

logger = logging.getLogger(__name__)

class QueryPlanner:
    def __init__(self, agentic_memory: AgenticMemory, llm_controller: Optional[LLMController] = None):
        """
        Initializes the QueryPlanner.

        Args:
            agentic_memory: An instance of AgenticMemory for memory interaction.
            llm_controller: An optional LLMController instance if the planner needs to use LLMs
                            for query understanding or strategy generation.
        """
        self.agentic_memory = agentic_memory
        self.llm_controller = llm_controller # May not be used in initial version

    def simple_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a straightforward content-based search using AgenticMemory.
        This is similar to the old 'search' method.
        """
        logger.debug(f"QueryPlanner performing simple search: '{query}', k={k}")
        try:
            return self.agentic_memory.search_notes_by_content(query, k)
        except Exception as e:
            logger.error(f"Error during simple_search in QueryPlanner: {e}", exc_info=True)
            return []

    def agentic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a more complex search, potentially involving linked memories.
        This adapts logic from the old 'search_agentic' method.
        The current AgenticMemory.search_notes_by_content provides scored results.
        This method could extend that by explicitly fetching and including linked notes.
        """
        logger.debug(f"QueryPlanner performing agentic search: '{query}', k={k}")
        try:
            # 1. Get initial search results from AgenticMemory
            # This method already returns rich dicts with score, content, metadata etc.
            initial_results = self.agentic_memory.search_notes_by_content(query, k)

            memories_found = []
            seen_ids = set()

            for res in initial_results:
                if res['id'] not in seen_ids:
                    res['is_neighbor'] = False # Mark that this is a directly retrieved note
                    memories_found.append(res)
                    seen_ids.add(res['id'])

            # 2. Augment with linked memories (neighbors)
            # We need to iterate over a copy if we modify memories_found inside loop,
            # but here we are fetching notes based on links from initial_results.

            # Limit how many neighbors we add to stay within k overall, or a separate limit
            # For simplicity, let's say we try to fill up to k results.

            # Create a list of notes that were directly found to iterate for links
            notes_to_check_links = [res['id'] for res in initial_results]

            for note_id_to_check in notes_to_check_links:
                if len(memories_found) >= k:
                    break # Stop if we already have k results

                # Get the full MemoryNote object to access its links
                # The agentic_memory.get() method loads from retriever if not in cache
                # and updates access stats.
                current_note_obj = self.agentic_memory.get(note_id_to_check)

                if current_note_obj and current_note_obj.links:
                    for link_id in current_note_obj.links:
                        if len(memories_found) >= k:
                            break
                        if link_id not in seen_ids:
                            neighbor_note_obj = self.agentic_memory.get(link_id)
                            if neighbor_note_obj:
                                memories_found.append({
                                    'id': neighbor_note_obj.id,
                                    'content': neighbor_note_obj.content,
                                    'context': neighbor_note_obj.context,
                                    'keywords': neighbor_note_obj.keywords,
                                    'tags': neighbor_note_obj.tags,
                                    'timestamp': neighbor_note_obj.timestamp,
                                    'category': neighbor_note_obj.category,
                                    'is_neighbor': True, # Mark as a linked neighbor
                                    'score': 0.0 # Neighbors don't have a direct query score
                                })
                                seen_ids.add(link_id)

            return memories_found[:k] # Ensure we don't exceed k

        except Exception as e:
            logger.error(f"Error during agentic_search in QueryPlanner: {e}", exc_info=True)
            return []

    # Future methods could involve LLM-based query decomposition or multi-step retrieval
    # def answer_question_with_llm(self, question: str, k: int = 5) -> Dict[str, Any]:
    #     if not self.llm_controller:
    #         logger.warning("LLMController not provided to QueryPlanner. Cannot answer question with LLM.")
    #         return {"answer": "LLM support not configured.", "sources": []}
    #
    #     # 1. Understand the question (optional, could be simple keyword extraction or complex parsing)
    #     # 2. Formulate a search query (or multiple)
    #     search_query = question # Simplistic for now
    #     retrieved_notes = self.agentic_search(search_query, k)
    #
    #     # 3. Synthesize an answer using LLM based on retrieved notes
    #     context_for_llm = ""
    #     for note in retrieved_notes:
    #         context_for_llm += f"ID: {note['id']}\nContent: {note['content']}\nContext: {note.get('context', '')}\n\n"
    #
    #     prompt = f"Based on the following information, answer the question: {question}\n\nInformation:\n{context_for_llm}"
    #     answer = self.llm_controller.get_completion(prompt, stage="AnswerSynthesis")
    #
    #     return {"answer": answer, "sources": retrieved_notes}

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single memory by its ID and formats it as a dictionary."""
        logger.debug(f"QueryPlanner getting memory by id: {memory_id}")
        note_obj = self.agentic_memory.get(memory_id)
        if note_obj:
            return {
                'id': note_obj.id,
                'content': note_obj.content,
                'context': note_obj.context,
                'keywords': note_obj.keywords,
                'tags': note_obj.tags,
                'timestamp': note_obj.timestamp,
                'category': note_obj.category,
                'links': note_obj.links,
                'retrieval_count': note_obj.retrieval_count,
                'last_accessed': note_obj.last_accessed,
                'evolution_history': note_obj.evolution_history
            }
        return None
