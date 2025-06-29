import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from llm_controller import LLMController
from llm_interaction_logger import LLMInteractionLogger # Added import
from retrievers import ChromaRetriever

from memory.note import MemoryNote
from memory.analyzer import Analyzer
from memory.evolver import Evolver
# memory.utils._extract_json_from_response is used by Analyzer and Evolver,
# but not directly by AgenticMemorySystem after refactoring.
# If other methods in AgenticMemorySystem were to need it, it should be imported here.
# from memory.utils import _extract_json_from_response

logger = logging.getLogger(__name__)


def _note_to_metadata(note: MemoryNote) -> Dict:
    """将MemoryNote对象转换为ChromaDB所需的元数据字典。"""
    return {
        "id": note.id,
        "content": note.content,
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


class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.

    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction (via Analyzer)
    - Memory evolution and relationship management (via Evolver)
    - Hybrid search capabilities
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 db_path: Optional[str] = None):
        """Initialize the memory system.

        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            db_path: Path to the database directory (Optional)
        """
        print("Initializing AgenticMemorySystem...")
        self.memories: Dict[str, MemoryNote] = {}
        self.model_name = model_name
        self.db_path = db_path

        if self.db_path is None:
            self.db_path = "./chroma_db"
            logger.warning(f"No database path provided. Using default path: {self.db_path}")
        print(f"DB path: {self.db_path}")

        self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name,
                                         db_path=self.db_path)
        print(f"Finish chroma retriever initialization")
        print(f"Initializing memory system with model: {self.model_name}")
        # Create and inject LLMInteractionLogger
        llm_interaction_logger_instance = LLMInteractionLogger() # Instantiate the logger

        self.llm_controller = LLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            logger=llm_interaction_logger_instance # Pass the logger instance
        )

        self.analyzer = Analyzer(llm_controller=self.llm_controller)
        self.evolver = Evolver(
            llm_controller=self.llm_controller,
            memories=self.memories,
            find_related_memories_callable=self.find_related_memories,
            update_memory_callable=self.update  # <--- 将 self.update 方法传递给 Evolver
        )

        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: Optional[str] = None, **kwargs) -> str:
        """
        添加一个新的记忆笔记，对其进行分析，并处理其演化。

        Args:
            content (str): 笔记的主要内容。
            time (Optional[str]): 可选的时间戳。
            **kwargs: 其他传递给MemoryNote的元数据。

        Returns:
            str: 创建的笔记的ID。
        """
        # 1. 使用分析器获取初始元数据
        analysis_results = self.analyzer.analyze_content(content)

        # 2. 准备创建MemoryNote所需的所有属性
        note_kwargs = {
            'keywords': kwargs.get('keywords', analysis_results.get('keywords', [])),
            'context': kwargs.get('context', analysis_results.get('context', "General")),
            'tags': kwargs.get('tags', analysis_results.get('tags', [])),
            'timestamp': time if time is not None else None
        }
        # 过滤掉值为None的键
        note_kwargs = {k: v for k, v in note_kwargs.items() if v is not None}

        note = MemoryNote(content=content, **note_kwargs)

        # 在处理演化之前，必须将新笔记先放入 self.memories
        # 这样 Evolver 才能在需要时（如添加反向链接）找到它。
        self.memories[note.id] = note

        # 3. 调用Evolver处理记忆演化
        # 这个方法现在会处理所有逻辑，包括更新邻居和持久化
        evo_label, updated_note = self.evolver.evolve_memory(note)

        # 4. 确保内存中的笔记对象是最新的
        self.memories[updated_note.id] = updated_note

        # 5. 将最终状态的新笔记持久化到ChromaDB
        # 注意：邻居的更新已经在 evolve_memory 内部通过调用 self.update() 完成了
        metadata = _note_to_metadata(updated_note)
        self.retriever.add_document(updated_note.content, metadata, updated_note.id)

        # 6. 检查是否需要进行内存整理
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        return updated_note.id

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset ChromaDB collection
        self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name,
                                         db_path=self.db_path)

        # Re-add all memory documents with their complete metadata
        for memory_id, memory in self.memories.items(): # Iterate through self.memories
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags
            }
            self.retriever.add_document(memory.content, metadata, memory.id)

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        Find related memories using ChromaDB retrieval and format them for the Evolver.

        This method is passed to the Evolver instance.

        Args:
            query: The text to find related memories for.
            k: The number of related memories to find.

        Returns:
            A tuple containing:
            - A formatted string of related memories for the LLM prompt, including their IDs.
            - A list of the actual memory note IDs (UUIDs).
        """
        if not self.memories:  # Check if self.memories is empty
            return "", []

        try:
            # 1. 从ChromaDB检索结果
            results = self.retriever.search(query, k)

            # 检查结果是否有效
            if not results or not results.get('ids') or not results['ids'][0]:
                return "", []

            # 2. 直接从检索结果中获取真实的UUID列表
            # 这解决了返回错误索引的问题。
            neighbor_ids = results['ids'][0]

            # 3. 构建提供给LLM的文本，并清晰地包含每个记忆的真实ID
            # 这将确保LLM在其响应中返回正确的ID。
            memory_parts = []
            for i, doc_id in enumerate(neighbor_ids):
                if doc_id in self.memories:
                    note = self.memories[doc_id]
                    # 使用f-string和清晰的标签来格式化，提高LLM的理解能力
                    memory_parts.append(
                        f"ID: {note.id}\n"
                        f"Content: {note.content}\n"
                        f"Context: {note.context}\n"
                        f"Tags: {note.tags}"
                    )

            # 用分隔符连接每个记忆的信息，使其更易于解析
            memory_str = "\n---\n".join(memory_parts)

            # 4. 返回格式化后的字符串和真实的UUID列表
            return memory_str, neighbor_ids

        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}", exc_info=True)
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""

        results = self.retriever.search(query, k)
        memory_str = ""

        if 'ids' in results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    memory_str += (f"talk start time:{metadata.get('timestamp', '')}\t"
                                   f"memory content: {metadata.get('content', '')}\t"
                                   f"memory context: {metadata.get('context', '')}\t"
                                   f"memory keywords: {str(metadata.get('keywords', []))}\t"
                                   f"memory tags: {str(metadata.get('tags', []))}\n")

                    # Retrieve full note for links, as links might not be in Chroma metadata
                    # or could be stale if not updated there.
                    current_note = self.memories.get(doc_id)
                    if current_note:
                        links = current_note.links
                        j = 0
                        for link_id in links: # Iterate over actual link IDs
                            if link_id in self.memories and j < k: # Check if linked memory exists
                                neighbor = self.memories[link_id]
                                memory_str += (f"talk start time:{neighbor.timestamp}\t"
                                               f"memory content: {neighbor.content}\t"
                                               f"memory context: {neighbor.context}\t"
                                               f"memory keywords: {str(neighbor.keywords)}\t"
                                               f"memory tags: {str(neighbor.tags)}\n")
                                j += 1
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.memories:
            return False

        note = self.memories[memory_id]
        updated_fields = False
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                updated_fields = True

        if updated_fields:
            # If content is updated, re-analyze for keywords, context, tags?
            # Current logic only updates fields directly.
            # For simplicity, we'll assume if 'content' is in kwargs, user provides new metadata too,
            # or relies on existing metadata. A more robust system might re-analyze.

            # Update last_accessed time
            note.last_accessed = datetime.now().strftime("%Y%m%d%H%M")

            metadata = {
                "id": note.id, "content": note.content, "keywords": note.keywords,
                "links": note.links, "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp, "last_accessed": note.last_accessed,
                "context": note.context, "evolution_history": note.evolution_history,
                "category": note.category, "tags": note.tags
            }
            self.retriever.delete_document(memory_id) # Delete old version
            self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id) # Add new
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            self.retriever.delete_document(memory_id)
            del self.memories[memory_id]
            return True
        return False

    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, k)
        # Check if results are valid and have the expected structure
        if results and 'ids' in results and results['ids'] and \
           'distances' in results and results['distances'] and \
           len(results['ids'][0]) == len(results['distances'][0]):
            return [{'id': doc_id, 'score': score}
                    for doc_id, score in zip(results['ids'][0], results['distances'][0])]
        return []


    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        search_results = self.retriever.search(query, k)
        memories_found = []

        if search_results and 'ids' in search_results and search_results['ids'][0]:
            for i, doc_id in enumerate(search_results['ids'][0]):
                memory = self.memories.get(doc_id)
                if memory:
                    # Ensure score is available and correctly accessed
                    score = 0.0
                    if 'distances' in search_results and search_results['distances'] and \
                       i < len(search_results['distances'][0]):
                        score = search_results['distances'][0][i]

                    memories_found.append({
                        'id': doc_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': score
                    })
        return memories_found[:k] # Ensure only k results are returned

    # _search seems to be a duplicate or an alternative version of search.
    # For this refactoring, I'll keep the 'search' method as the primary one
    # and assume _search might be for internal variations or future use.
    # If it's intended to be the main one, then 'search' should call '_search'.
    # Based on current usage (e.g. no internal calls to _search shown),
    # I'll leave it as is but note the potential redundancy.

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.memories:
            return []

        try:
            results = self.retriever.search(query, k)
            memories_found = []
            seen_ids = set()

            if not (results and 'ids' in results and results['ids'] and results['ids'][0]):
                return []

            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue

                # Ensure metadatas exist and index is valid
                if not ('metadatas' in results and results['metadatas'] and
                        i < len(results['metadatas'][0])):
                    continue

                metadata = results['metadatas'][0][i]
                # It's safer to get the full note from self.memories for consistency
                current_note = self.memories.get(doc_id)
                if not current_note:
                    # If somehow not in self.memories, use metadata but log a warning
                    logger.warning(f"Memory ID {doc_id} found in retriever but not in self.memories.")
                    # Fallback to metadata from retriever, but this indicates potential inconsistency
                    display_content = metadata.get('content', '')
                    display_context = metadata.get('context', '')
                    display_keywords = metadata.get('keywords', [])
                    display_tags = metadata.get('tags', [])
                    display_timestamp = metadata.get('timestamp', '')
                    display_category = metadata.get('category', 'Uncategorized')
                    note_links = metadata.get('links', []) # links might not be in metadata
                else:
                    display_content = current_note.content
                    display_context = current_note.context
                    display_keywords = current_note.keywords
                    display_tags = current_note.tags
                    display_timestamp = current_note.timestamp
                    display_category = current_note.category
                    note_links = current_note.links


                memory_dict = {
                    'id': doc_id,
                    'content': display_content,
                    'context': display_context,
                    'keywords': display_keywords,
                    'tags': display_tags,
                    'timestamp': display_timestamp,
                    'category': display_category,
                    'is_neighbor': False
                }

                if 'distances' in results and results['distances'] and i < len(results['distances'][0]):
                    memory_dict['score'] = results['distances'][0][i]

                memories_found.append(memory_dict)
                seen_ids.add(doc_id)

            # Add linked memories (neighbors)
            neighbor_count = 0
            # Iterate over a copy if modifying list during iteration (not the case here, but good practice)
            for memory_data in list(memories_found):
                if neighbor_count >= k:
                    break

                # Get links from the original note if available, else from metadata (less reliable for links)
                # The 'doc_id' for the current 'memory_data' is memory_data['id']
                original_note = self.memories.get(memory_data['id'])
                links_to_check = original_note.links if original_note else []


                for link_id in links_to_check:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories_found.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1

            return memories_found[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def shutdown(self):
        """Shuts down the memory system and its components (e.g., retriever)."""
        if hasattr(self.retriever, 'shutdown'):
            self.retriever.shutdown()
        # Potentially shutdown LLMController if it holds resources, e.g., connections
        # if hasattr(self.llm_controller, 'shutdown'):
        #     self.llm_controller.shutdown()
        logger.info("AgenticMemorySystem shutdown complete.")

# Example of how AgenticMemorySystem might be initialized (outside the class)
if __name__ == '__main__':
    # This is for demonstration; actual API keys and paths should be handled securely
    # and passed appropriately, perhaps via environment variables or a config file.
    # logging.basicConfig(level=logging.INFO)
    # memory_system = AgenticMemorySystem(api_key="YOUR_API_KEY_HERE_IF_NEEDED")
    # new_note_id = memory_system.add_note("This is a test note about AI and memory.",
    #                                      category="Technology",
    #                                      tags=["AI", "testing", "refactor"])
    # logger.info(f"Added new note with ID: {new_note_id}")
    # search_results = memory_system.search("AI memory", k=2)
    # logger.info(f"Search results: {json.dumps(search_results, indent=2)}")
    # memory_system.shutdown()
    pass # Placeholder for example usage
