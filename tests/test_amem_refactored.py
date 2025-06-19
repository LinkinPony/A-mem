import unittest
import os
import shutil
import logging
from typing import Dict, Any

# Configure logging for tests
logging.basicConfig(level=logging.INFO) # Show info for debugging tests
logger = logging.getLogger(__name__)

# New imports for the refactored system
from memory import ChromaRetriever, AgenticMemory, QueryPlanner, MemoryNote, BaseRetriever
from llm_controller import LLMController
from llm_interaction_logger import LLMInteractionLogger

# Configuration for tests (adjust as needed)
TEST_DB_PATH = "./test_chroma_db"
TEST_MODEL_NAME = "all-MiniLM-L6-v2" # Use a model accessible in test env
TEST_COLLECTION_NAME = "test_memories"

# Mock LLMController for tests where LLM interaction is not the focus
class MockLLMController(LLMController):
    def __init__(self, backend="mock", model="mock_model", api_key=None, logger_instance=None):
        super().__init__(backend, model, api_key, logger_instance)
        self.mock_responses = {}

    def set_mock_response(self, prompt_key_part, response):
        # Allow setting mock responses based on parts of the prompt
        self.mock_responses[prompt_key_part] = response

    def get_completion(self, prompt: str, response_format: Dict = None, stage: str = "test") -> str:
        for key, resp in self.mock_responses.items():
            if key in prompt:
                if isinstance(resp, dict) and response_format and response_format.get("type") == "json_object":
                    import json
                    return json.dumps(resp)
                return str(resp) # Ensure string return for general case

        # Default mock responses for Analyzer and Evolver if not specifically set
        if "Generate a structured analysis" in prompt: # Analyzer prompt
            return '{"keywords": ["mock_keyword"], "context": "mock_context", "tags": ["mock_tag"]}'
        if "You are an AI memory evolution agent" in prompt: # Evolver prompt
            return '{"should_evolve": false, "actions": []}'

        logger.warning(f"MockLLMController received unmocked prompt for stage '{stage}': {prompt[:100]}...")
        return "mocked LLM response"

class TestRefactoredAMemSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure no old DB exists from previous runs
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH)
        os.makedirs(TEST_DB_PATH, exist_ok=True)

        cls.llm_logger = LLMInteractionLogger(log_to_console=False) # Disable console log for tests unless debugging
        cls.llm_controller = MockLLMController(logger_instance=cls.llm_logger)

        # Setup mock responses for typical interactions
        cls.llm_controller.set_mock_response(
            "Generate a structured analysis", # Analyzer
            {"keywords": ["test", "mock"], "context": "Test context", "tags": ["testing"]}
        )
        cls.llm_controller.set_mock_response(
            "You are an AI memory evolution agent", # Evolution
            {"should_evolve": True, "actions": [
                # Example: Strengthen action (ensure target_memory_id is valid in test setup)
                # {"action_type": "strengthen", "target_memory_id": "mock_neighbor_id", "new_tags_for_current_note": ["evolved_tag"]}
            ]} # Default to no evolution if no specific test
        )


    @classmethod
    def tearDownClass(cls):
        # Clean up the test database
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH)

    def setUp(self):
        # This method is called before each test function.
        # Re-initialize components for each test to ensure isolation.
        # Important: ChromaRetriever's shutdown now calls client.reset() which clears the DB.
        # So, for each test, we get a fresh DB if we instantiate it per test.
        # Or, manage DB clearing carefully.
        # For simplicity, let's create a new retriever for each test, which means new collection.

        # Clean DB directory before each test to ensure isolation if retriever persists across tests
        # If retriever is per test, this is less critical but good for safety.
        if os.path.exists(TEST_DB_PATH): # Ensure clean slate for each test
            # This can be slow. A better way might be to use different collection names per test
            # or ensure client.reset() is effective.
            # self.retriever.shutdown() will call client.reset()
            # For now, let's assume each test method will manage its retriever instance or clean up.
            pass

        self.retriever = ChromaRetriever(
            collection_name=f"{TEST_COLLECTION_NAME}_{self._testMethodName}", # Unique collection per test
            model_name=TEST_MODEL_NAME,
            db_path=TEST_DB_PATH
        )
        self.agentic_memory = AgenticMemory(retriever=self.retriever, llm_controller=self.llm_controller)
        self.query_planner = QueryPlanner(agentic_memory=self.agentic_memory, llm_controller=self.llm_controller)

    def tearDown(self):
        # Called after each test method.
        # Shutdown retriever to clear the specific test collection
        if hasattr(self, 'retriever') and self.retriever:
            self.retriever.shutdown() # This should call client.reset() and clear the collection/DB for Chroma

    def test_01_retriever_add_and_retrieve(self):
        logger.info("Running test_01_retriever_add_and_retrieve")
        doc_id = "test_doc_1"
        document = "This is a test document for ChromaRetriever."
        metadata = {"source": "test", "type": "document"}

        self.retriever.add(document, metadata, doc_id)
        retrieved = self.retriever.retrieve_by_id(doc_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['document'], document)
        self.assertEqual(retrieved['metadata']['source'], "test")

        # Test search
        search_results = self.retriever.search("test document", k=1)
        self.assertIn(doc_id, search_results['ids'])

    def test_02_agentic_memory_add_and_get(self):
        logger.info("Running test_02_agentic_memory_add_and_get")
        content = "Test memory note about AI ethics."
        note_id = self.agentic_memory.add(content, category="AI Safety", tags=["ethics", "ai"])

        self.assertIsNotNone(note_id)

        retrieved_note = self.agentic_memory.get(note_id)
        self.assertIsNotNone(retrieved_note)
        self.assertEqual(retrieved_note.content, content)
        self.assertEqual(retrieved_note.category, "AI Safety")
        self.assertIn("ethics", retrieved_note.tags)

        # Verify it's in the retriever too
        retrieved_from_db = self.retriever.retrieve_by_id(note_id)
        self.assertIsNotNone(retrieved_from_db)
        self.assertEqual(retrieved_from_db['document'], content)

    def test_03_agentic_memory_update(self):
        logger.info("Running test_03_agentic_memory_update")
        content = "Initial content for update test."
        note_id = self.agentic_memory.add(content, tags=["initial"])

        updated_content = "Updated content for the note."
        updated_tags = ["updated", "test"]
        success = self.agentic_memory.update(note_id, content=updated_content, tags=updated_tags)
        self.assertTrue(success)

        updated_note = self.agentic_memory.get(note_id)
        self.assertEqual(updated_note.content, updated_content)
        self.assertEqual(updated_note.tags, updated_tags)

    def test_04_agentic_memory_delete(self):
        logger.info("Running test_04_agentic_memory_delete")
        content = "Content to be deleted."
        note_id = self.agentic_memory.add(content)

        self.assertIsNotNone(self.agentic_memory.get(note_id), "Note should exist before delete")

        success = self.agentic_memory.delete(note_id)
        self.assertTrue(success)
        self.assertIsNone(self.agentic_memory.get(note_id), "Note should be deleted from memory")

        retrieved_from_db = self.retriever.retrieve_by_id(note_id)
        self.assertIsNone(retrieved_from_db, "Note should be deleted from retriever")

    def test_05_query_planner_simple_search(self):
        logger.info("Running test_05_query_planner_simple_search")
        self.agentic_memory.add("Mars is a red planet.", category="Space")
        note_id_sun = self.agentic_memory.add("The Sun is a star.", category="Space")

        search_results = self.query_planner.simple_search("bright star", k=1)
        self.assertGreater(len(search_results), 0, "Should find at least one result")
        # Depending on embedding model, "sun" or "star" should be found.
        # Check if the most relevant ID is among those added.
        self.assertTrue(any(r['id'] == note_id_sun for r in search_results), "Sun note not found in search for 'bright star'")


    def test_06_memory_evolution_mocked(self):
        logger.info("Running test_06_memory_evolution_mocked")
        # Setup a specific mock response for evolution for this test
        # This assumes an existing note 'neighbor_for_evo_test' that the new note can link to.
        neighbor_id = self.agentic_memory.add("This is a neighbor note for evolution testing.", tags=["neighbor"])

        self.llm_controller.set_mock_response(
            "You are an AI memory evolution agent",
            {
                "should_evolve": True,
                "actions": [{
                    "action_type": "strengthen",
                    "target_memory_id": neighbor_id,
                    "new_tags_for_current_note": ["evolved_tag", "linked"]
                }]
            }
        )

        new_note_content = "A new note that should trigger evolution and link to the neighbor."
        new_note_id = self.agentic_memory.add(new_note_content)

        evolved_note = self.agentic_memory.get(new_note_id)
        self.assertIn("evolved_tag", evolved_note.tags)
        self.assertIn(neighbor_id, evolved_note.links)

        # Check backlink on neighbor
        neighbor_note = self.agentic_memory.get(neighbor_id)
        self.assertIn(new_note_id, neighbor_note.links)


    def test_07_query_planner_agentic_search(self):
        logger.info("Running test_07_query_planner_agentic_search")
        # Add a base note
        base_content = "Information about topic X."
        base_id = self.agentic_memory.add(base_content, tags=["topic_x_base"])

        # Add a linked note. For this, we need to simulate evolution or manually link.
        # Let's manually update the base_note to link to another.
        linked_content = "Details related to topic X."
        linked_id = self.agentic_memory.add(linked_content, tags=["topic_x_detail"])

        # Manually create a link for testing agentic_search's link traversal
        base_note = self.agentic_memory.get(base_id)
        base_note.links.append(linked_id)
        self.agentic_memory.update(base_id, links=base_note.links)
        # Also add backlink for consistency, though agentic_search might not rely on it directly
        linked_note = self.agentic_memory.get(linked_id)
        linked_note.links.append(base_id)
        self.agentic_memory.update(linked_id, links=linked_note.links)

        # Mock LLM response for evolution during these adds if they trigger it, to avoid side effects
        # This is a general mock; specific tests might need more tailored responses
        self.llm_controller.set_mock_response("You are an AI memory evolution agent", {"should_evolve": False, "actions": []})


        # Search for the base content
        search_results = self.query_planner.agentic_search("topic X", k=2)

        self.assertTrue(len(search_results) >= 1, "Agentic search should return at least the base note.")

        found_ids = [r['id'] for r in search_results]
        self.assertIn(base_id, found_ids, "Base note not found in agentic search.")

        # Check if the linked note is also returned
        # This depends on the agentic_search logic correctly finding and including linked notes
        is_linked_note_present = any(r['id'] == linked_id and r.get('is_neighbor', False) for r in search_results)

        # If k=2 and both are relevant, both should be there.
        # If only base_id is directly found, linked_id should be brought in as a neighbor.
        # This assertion might need adjustment based on exact search and ranking logic.
        if len(search_results) > 1: # If more than one result, check for linked note
             self.assertIn(linked_id, found_ids, "Linked note not found or not marked as neighbor in agentic search.")
        logger.info(f"Agentic search results: {search_results}")


if __name__ == '__main__':
    unittest.main()
