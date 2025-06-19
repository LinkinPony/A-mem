import json
import logging
from typing import List, Dict, Tuple, Optional, Callable

from llm_controller import LLMController
from memory.note import MemoryNote
from memory.utils import _extract_json_from_response

logger = logging.getLogger(__name__)

class Evolver:
    _evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''

    def __init__(self,
                 llm_controller: LLMController,
                 memories: Dict[str, MemoryNote],
                 find_related_memories_callable: Callable[[str, int], Tuple[str, List[int]]]):
        self.llm_controller = llm_controller
        self.memories = memories
        self.find_related_memories = find_related_memories_callable

    def evolve_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.

        Args:
            note: The memory note to process

        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note

        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note

            # Format neighbors for LLM - in this case, neighbors_text is already formatted

            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices)
            )

            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections",
                                         "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )

                # Clean the LLM response before parsing and use .get() for safe access.
                clean_json_str = _extract_json_from_response(response)
                if not clean_json_str:
                    logger.error("Could not extract valid JSON from LLM response for evolution. Response: %s", response)
                    return False, note

                response_json = json.loads(clean_json_str)
                should_evolve = response_json.get("should_evolve", False)

                if should_evolve:
                    actions = response_json.get("actions", [])
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json.get("suggested_connections", [])
                            new_tags = response_json.get("tags_to_update", [])
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                            new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                            noteslist = list(self.memories.values())
                            notes_id = list(self.memories.keys())

                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                # Skip if we don't have enough neighbors
                                if i >= len(indices):
                                    continue

                                tag = new_tags_neighborhood[i]
                                if i < len(new_context_neighborhood):
                                    context = new_context_neighborhood[i]
                                else:
                                    # Since indices are just numbers now, we need to find the memory
                                    # In memory list using its index number
                                    if i < len(noteslist):
                                        context = noteslist[i].context
                                    else:
                                        continue

                                # Get index from the indices list
                                if i < len(indices):
                                    memorytmp_idx = indices[i]
                                    # Make sure the index is valid
                                    if memorytmp_idx < len(noteslist):
                                        notetmp = noteslist[memorytmp_idx]
                                        notetmp.tags = tag
                                        notetmp.context = context
                                        # Make sure the index is valid
                                        if memorytmp_idx < len(notes_id):
                                            self.memories[notes_id[memorytmp_idx]] = notetmp

                return should_evolve, note

            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note

        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
