import logging
import os
import json
import shutil
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

from memory import ChromaRetriever, AgenticMemory, QueryPlanner, MemoryNote
from llm_controller import LLMController
from llm_interaction_logger import LLMInteractionLogger

# Configuration
DB_PATH_ADV = "./adv_example_chroma_db"
COLLECTION_NAME_ADV = "adv_example_memories"
MODEL_NAME_ADV = "all-MiniLM-L6-v2"

LLM_BACKEND_ADV = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL_ADV = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY_ADV = os.getenv("OPENAI_API_KEY")


def setup_components():
    logger.info("Setting up advanced example components...")
    llm_logger_adv = LLMInteractionLogger(log_file="adv_llm_interactions.log", log_to_console=True)

    if LLM_BACKEND_ADV == "openai" and not OPENAI_API_KEY_ADV:
        logger.error("OPENAI_API_KEY is not set for advanced example. LLM features will be impacted.")
        # Fallback to a mock or dummy for the example to run without OpenAI key
        llm_controller_adv = LLMController(backend="mock", model="mock-adv-model", api_key="DUMMY_KEY_ADV", logger=llm_logger_adv)
    else:
        llm_controller_adv = LLMController(
            backend=LLM_BACKEND_ADV,
            model=LLM_MODEL_ADV,
            api_key=OPENAI_API_KEY_ADV,
            logger=llm_logger_adv
        )

    retriever_adv = ChromaRetriever(
        collection_name=COLLECTION_NAME_ADV,
        model_name=MODEL_NAME_ADV,
        db_path=DB_PATH_ADV
    )
    agentic_memory_adv = AgenticMemory(retriever=retriever_adv, llm_controller=llm_controller_adv)
    query_planner_adv = QueryPlanner(agentic_memory=agentic_memory_adv, llm_controller=llm_controller_adv)

    logger.info("Advanced example components initialized.")
    return retriever_adv, agentic_memory_adv, query_planner_adv, llm_controller_adv

def run_advanced_example():
    logger.info("Starting A-Mem Refactored Advanced Example...")

    retriever, agentic_memory, query_planner, llm_c = setup_components()

    # 1. Add a series of related memories to showcase evolution (if LLM is capable)
    logger.info("\n--- Adding and Evolving Memories ---")
    # For evolution to be predictable in an example without a real LLM,
    # one would typically use a MockLLMController with predefined responses.
    # Here, we rely on the actual LLM or the mock's default if key is missing.

    try:
        m1_text = "The history of computing began with mechanical calculators."
        m1_id = agentic_memory.add(m1_text, category="History", tags=["computing", "calculators"])
        logger.info(f"Added: '{m1_text[:30]}...' (ID: {m1_id})")

        m2_text = "Charles Babbage designed the Analytical Engine, a conceptual forerunner to modern computers."
        m2_id = agentic_memory.add(m2_text, category="History", tags=["computing", "analytical_engine", "babbage"])
        logger.info(f"Added: '{m2_text[:30]}...' (ID: {m2_id})")

        # This note should ideally link to m1_id or m2_id if LLM-based evolution works as expected
        m3_text = "Ada Lovelace is considered the first computer programmer for her work on the Analytical Engine."
        m3_id = agentic_memory.add(m3_text, category="History", tags=["computing", "ada_lovelace", "programmer"])
        logger.info(f"Added: '{m3_text[:30]}...' (ID: {m3_id})")

        # Check links for m3 (Ada Lovelace note)
        ada_note = agentic_memory.get(m3_id)
        if ada_note and ada_note.links:
            logger.info(f"Memory '{ada_note.content[:30]}...' (ID: {m3_id}) has links: {ada_note.links}")
            for link_id in ada_note.links:
                linked_note = agentic_memory.get(link_id)
                logger.info(f"  - Linked to: '{linked_note.content[:30]}...' (ID: {link_id})")
        else:
            logger.info(f"Memory '{ada_note.content[:30]}...' (ID: {m3_id}) has no links after add. (Evolution might not have occurred or LLM is mocked/unavailable).")

    except Exception as e:
        logger.error(f"Error during advanced memory addition/evolution: {e}", exc_info=True)


    # 2. Perform a more nuanced search
    logger.info("\n--- Nuanced Search (Agentic Search) ---")
    query = "early female computer pioneers"
    logger.info(f"Searching for: '{query}'")
    results = query_planner.agentic_search(query, k=3)
    if results:
        logger.info("Search results:")
        for r in results:
            logger.info(f"  ID: {r['id']}, Score: {r.get('score', 'N/A')}, Content: {r['content'][:60]}..., Neighbor: {r.get('is_neighbor', False)}")
    else:
        logger.info("No results found for nuanced search.")

    # 3. Inspecting a specific memory and its potential evolution
    logger.info("\n--- Inspecting a specific memory ---")
    if 'm3_id' in locals() and m3_id:
        inspected_note = query_planner.get_memory_by_id(m3_id) # Uses agentic_memory.get()
        if inspected_note:
            logger.info(f"Details for memory ID {m3_id}:")
            logger.info(json.dumps(inspected_note, indent=2))
        else:
            logger.warning(f"Could not retrieve memory {m3_id} for inspection.")
    else:
        logger.info("m3_id not available for inspection.")

    # 4. Using LLM for a task (if QueryPlanner had such a method and LLM is configured)
    # This part is hypothetical based on the QueryPlanner's commented-out `answer_question_with_llm`
    # logger.info("\n--- LLM-based Question Answering (Hypothetical) ---")
    # if hasattr(query_planner, 'answer_question_with_llm') and llm_c.backend != "mock":
    #     qa_query = "Who was considered the first computer programmer?"
    #     logger.info(f"Asking LLM (via QueryPlanner): '{qa_query}'")
    #     answer_obj = query_planner.answer_question_with_llm(qa_query, k=2)
    #     logger.info(f"LLM Answer: {answer_obj.get('answer')}")
    #     logger.info("Sources:")
    #     for src in answer_obj.get('sources', []):
    #         logger.info(f"  - ID: {src['id']}, Content: {src['content'][:40]}...")
    # else:
    #     logger.info("Skipping LLM-based QA example (method not implemented or LLM is mock/not configured).")


    logger.info("Shutting down advanced example retriever...")
    retriever.shutdown()
    logger.info("Advanced example finished.")

if __name__ == "__main__":
    # Clean up old DB for a fresh run
    if os.path.exists(DB_PATH_ADV):
        logger.info(f"Removing old database at {DB_PATH_ADV} for a fresh advanced example run.")
        try:
            shutil.rmtree(DB_PATH_ADV)
        except OSError as e:
            logger.error(f"Error removing directory {DB_PATH_ADV}: {e.strerror}")
    run_advanced_example()
