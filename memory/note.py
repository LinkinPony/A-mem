from typing import Optional, List
from datetime import datetime
import uuid

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.

    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 # --- START: MINIMAL CHANGE 2 of 4 (Type Hint Correction) ---
                 links: Optional[List[str]] = None,  # Corrected type hint from Dict to List[str]
                 # --- END: MINIMAL CHANGE 2 of 4 ---
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.

        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format యyyyMMDDHHMM
            last_accessed (Optional[str]): Last access time in format యyyyMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())

        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []

        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []
