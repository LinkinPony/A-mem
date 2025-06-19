import logging
import os
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Assuming the new classes are available via the memory package
from memory import ChromaRetriever, AgenticMemory, QueryPlanner, MemoryNote
from llm_controller import LLMController
from llm_interaction_logger import LLMInteractionLogger

# Configuration (consider moving to a config file or environment variables)
DB_PATH = "./example_chroma_db"
COLLECTION_NAME = "example_memories"
MODEL_NAME = "all-MiniLM-L6-v2" # Sentence transformer model

# LLM Configuration - replace with your actual details if not using mocks
# Ensure OPENAI_API_KEY is set in your environment or .env file
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai") # "openai" or "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") # e.g., "gpt-4o-mini" or your Ollama model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_example():
    logger.info("Starting A-Mem Refactored Example...")

    # 1. Initialize components
    llm_interaction_logger = LLMInteractionLogger(log_to_console=True) # Log LLM interactions

    # Ensure API key is provided if using OpenAI
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Please set it in your .env file or environment.")
        # You could raise an error here or use a mock/dummy LLM controller for the example
        # For this example, let's try to proceed but warn that LLM features will fail.
        # llm_controller = LLMController(backend="mock", model="mock", logger=llm_interaction_logger)
        print("Error: OPENAI_API_KEY not found. LLM functionalities will be limited or fail.")
        print("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")
        # Fallback or exit
        llm_controller = LLMController(backend="mock", model="mock-model-for-example", api_key=" DUMMY_KEY ", logger=llm_interaction_logger)

    else:
        llm_controller = LLMController(
            backend=LLM_BACKEND,
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            logger=llm_interaction_logger
        )

    retriever = ChromaRetriever(
        collection_name=COLLECTION_NAME,
        model_name=MODEL_NAME,
        db_path=DB_PATH
    )
    # Important: Ensure the DB is clean for a fresh example run, or manage collections.
    # retriever.client.reset() # This clears the ENTIRE database. Use with caution.
    # Or, delete and recreate the specific collection if supported, or use unique names.
    # For this example, we'll rely on Chroma's get_or_create_collection.

    agentic_memory = AgenticMemory(retriever=retriever, llm_controller=llm_controller)
    query_planner = QueryPlanner(agentic_memory=agentic_memory, llm_controller=llm_controller)

    logger.info("All components initialized.")

    # 2. Add some memories
    logger.info("Adding memories...")
    try:
        note_id1 = agentic_memory.add("The first memory is about a sunny day in Paris.", category="Travel", tags=["paris", "weather", "experience"])
        logger.info(f"Added memory 1 with ID: {note_id1}")

        note_id2 = agentic_memory.add("A second memory describes a recipe for pasta carbonara.", category="Food", tags=["recipe", "pasta", "italian"])
        logger.info(f"Added memory 2 with ID: {note_id2}")

        note_id3 = agentic_memory.add("Thinking about the future of artificial intelligence and its impact on society.", category="Technology", tags=["ai", "future", "society", "ethics"])
        logger.info(f"Added memory 3 with ID: {note_id3}")

    except Exception as e:
        logger.error(f"Error adding memories: {e}", exc_info=True)
        # If LLM calls fail due to API key issues, this is where it might show up.
        # The mock LLMController in tests is different from a real one here.

    # 3. Retrieve a memory by ID
    logger.info("\n--- Retrieving a memory by ID ---")
    if 'note_id1' in locals() and note_id1: # Check if note_id1 was successfully created
        retrieved_note_obj = agentic_memory.get(note_id1) # Returns MemoryNote object
        if retrieved_note_obj:
            logger.info(f"Retrieved memory by ID ({note_id1}):")
            logger.info(f"  Content: {retrieved_note_obj.content}")
            logger.info(f"  Category: {retrieved_note_obj.category}")
            logger.info(f"  Tags: {retrieved_note_obj.tags}")
            logger.info(f"  Links: {retrieved_note_obj.links}")
        else:
            logger.warning(f"Could not retrieve memory with ID: {note_id1}")
    else:
        logger.warning("note_id1 not available, skipping retrieval by ID.")


    # 4. Search for memories using QueryPlanner's simple search
    logger.info("\n--- Simple Search via QueryPlanner ---")
    search_query_simple = "italian food"
    logger.info(f"Searching for: '{search_query_simple}'")
    simple_search_results = query_planner.simple_search(search_query_simple, k=2)
    if simple_search_results:
        logger.info("Simple search results:")
        for res in simple_search_results:
            logger.info(f"  ID: {res['id']}, Score: {res.get('score', 'N/A')}, Content: {res['content'][:50]}...")
            logger.info(f"    Tags: {res.get('tags')}")
    else:
        logger.info("No results found for simple search.")

    # 5. Search for memories using QueryPlanner's agentic search
    logger.info("\n--- Agentic Search via QueryPlanner ---")
    search_query_agentic = "artificial intelligence ethics"
    logger.info(f"Searching for: '{search_query_agentic}'")
    agentic_search_results = query_planner.agentic_search(search_query_agentic, k=3)
    if agentic_search_results:
        logger.info("Agentic search results:")
        for res in agentic_search_results:
            logger.info(f"  ID: {res['id']}, Score: {res.get('score', 'N/A')}, Content: {res['content'][:60]}..., Neighbor: {res.get('is_neighbor', False)}")
            logger.info(f"    Tags: {res.get('tags')}")
    else:
        logger.info("No results found for agentic search.")

    # 6. Example of updating a memory (optional)
    logger.info("\n--- Updating a memory ---")
    if 'note_id1' in locals() and note_id1:
        update_success = agentic_memory.update(note_id1, content="The first memory is now about a rainy day in Paris, but it was still beautiful.", tags=["paris", "weather", "experience", "rainy_day_update"])
        if update_success:
            logger.info(f"Successfully updated memory {note_id1}.")
            updated_note = agentic_memory.get(note_id1)
            logger.info(f"  New content snippet: {updated_note.content[:50]}...")
            logger.info(f"  New tags: {updated_note.tags}")
        else:
            logger.warning(f"Failed to update memory {note_id1}")
    else:
        logger.warning("note_id1 not available, skipping update example.")


    # 7. Example of deleting a memory (optional)
    # logger.info("\n--- Deleting a memory ---")
    # if 'note_id2' in locals() and note_id2:
    #     delete_success = agentic_memory.delete(note_id2)
    #     if delete_success:
    #         logger.info(f"Successfully deleted memory {note_id2}.")
    #         self.assertIsNone(agentic_memory.get(note_id2), "Note should be deleted.")
    #     else:
    #         logger.warning(f"Failed to delete memory {note_id2}")
    # else:
    #    logger.warning("note_id2 not available, skipping delete example.")


    # Cleanup: Shutdown retriever (optional, depends on use case)
    # For this example, explicitly shutting down to release resources and clear DB if reset is implemented in shutdown.
    logger.info("Shutting down retriever...")
    retriever.shutdown()
    logger.info("Example finished.")

if __name__ == "__main__":
    # Clean up old DB if it exists for a fresh run of the example
    if os.path.exists(DB_PATH):
        logger.info(f"Removing old database at {DB_PATH} for a fresh example run.")
        try:
            shutil.rmtree(DB_PATH)
        except OSError as e:
            logger.error(f"Error removing directory {DB_PATH}: {e.strerror}")
            logger.warning("Please ensure the DB path is clear if you want a completely fresh example.")

    run_example()
