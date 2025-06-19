# memory/__init__.py

# Core components of the A-Mem system
from .core.agentic_memory import AgenticMemory
from .core.analyzer import Analyzer
from .note import MemoryNote

# Storage layer components
from .storage.base_retriever import BaseRetriever
from .storage.chroma_retriever import ChromaRetriever

# Strategy layer components
from .strategy.query_planner import QueryPlanner

# Utilities (if any are meant for public consumption)
# from .utils import some_utility_if_needed

__all__ = [
    "AgenticMemory",
    "Analyzer",
    "MemoryNote",
    "BaseRetriever",
    "ChromaRetriever",
    "QueryPlanner"
]

# Optional: Configure logging for the memory package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
