import json
import logging
from typing import List, Dict, Tuple, Callable, Union
from pydantic import BaseModel, Field

from llm_controller import LLMController
from memory.note import MemoryNote
from memory.utils import _extract_json_from_response

logger = logging.getLogger(__name__)


class StrengthenAction(BaseModel):
    action_type: str = Field(..., pattern="^strengthen$")
    target_memory_id: str = Field(..., description="要加强连接的目标邻近记忆的ID。")
    new_tags_for_current_note: List[str] = Field(..., description="为当前正在处理的记忆更新的标签列表。")


class UpdateNeighborAction(BaseModel):
    action_type: str = Field(..., pattern="^update_neighbor$")
    target_memory_id: str = Field(..., description="要更新的邻近记忆的ID。")
    new_context: str = Field(..., description="为目标邻近记忆生成的新上下文。")
    new_tags: List[str] = Field(..., description="为目标邻近记忆生成的新标签列表。")


class EvolutionDecision(BaseModel):
    should_evolve: bool = Field(..., description="是否应该演化当前记忆。")
    actions: List[Union[StrengthenAction, UpdateNeighborAction]] = Field(..., description="要执行的演化操作列表。")


class Evolver:
    _evolution_system_prompt = '''
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

    def __init__(self,
                 llm_controller: LLMController,
                 memories: Dict[str, MemoryNote],
                 find_related_memories_callable: Callable[[str, int], Tuple[str, List[str]]],
                 # 新增一个参数来接收系统的 update 方法
                 update_memory_callable: Callable):
        self.llm_controller = llm_controller
        self.memories = memories
        self.find_related_memories = find_related_memories_callable
        self.update_memory = update_memory_callable

    def evolve_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        if not self.memories:
            return False, note

        try:
            neighbors_text, neighbor_ids = self.find_related_memories(note.content, 5)
            if not neighbors_text or not neighbor_ids:
                return False, note

            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                note_id=note.id,
                nearest_neighbors_memories=neighbors_text,
            )

            response = self.llm_controller.get_completion(
                prompt,
                response_format={"type": "json_object", "schema": EvolutionDecision},
                stage="Memory Evolution"
            )

            clean_json_str = _extract_json_from_response(response)
            if not clean_json_str:
                logger.error("Could not extract valid JSON from LLM response. Response: %s", response)
                return False, note

            decision = EvolutionDecision.model_validate_json(clean_json_str)

            if decision.should_evolve:
                for action in decision.actions:
                    if isinstance(action, StrengthenAction):
                        if action.target_memory_id in neighbor_ids:
                            # 更新当前笔记 (这会在 add_note 的末尾被持久化)
                            note.links.append(action.target_memory_id)
                            note.tags = action.new_tags_for_current_note
                            logger.info(f"Strengthened note {note.id} with link to {action.target_memory_id}")

                            # **核心修复**：为邻居添加反向链接并立即持久化
                            neighbor_to_link_back = self.memories.get(action.target_memory_id)
                            if neighbor_to_link_back:
                                if note.id not in neighbor_to_link_back.links:
                                    # 创建一个新的链接列表并调用 update
                                    new_links = neighbor_to_link_back.links + [note.id]
                                    self.update_memory(neighbor_to_link_back.id, links=new_links)
                                    logger.info(f"Persisted backlink from {neighbor_to_link_back.id} to {note.id}")

                    elif isinstance(action, UpdateNeighborAction):
                        if action.target_memory_id in self.memories:
                            # **核心修复**：调用 update 方法来同时更新内存和数据库
                            self.update_memory(
                                action.target_memory_id,
                                context=action.new_context,
                                tags=action.new_tags
                            )
                            logger.info(f"Updated and persisted neighbor note {action.target_memory_id}")

            return decision.should_evolve, note

        except Exception as e:
            logger.error(f"Error in memory evolution: {str(e)}", exc_info=True)
            return False, note
