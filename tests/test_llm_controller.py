import unittest
from unittest.mock import patch, MagicMock
import os

# Assuming llm_controller.py is in the parent directory or PYTHONPATH is set up
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai # Corrected import
from llm_controller import GeminiController, LLMController # Added LLMController
# Removed: from google.genai import types
# Removed: from google.genai import errors as genai_errors
from llm_interaction_logger import LLMInteractionLogger # Added LLMInteractionLogger

class TestGeminiController(unittest.TestCase):

    @patch('os.getenv')
    @patch('google.generativeai.Client') # Corrected patch target
    def test_gemini_get_completion_json_output(self, mock_google_generativeai_Client, mock_os_getenv):
        # Arrange
        mock_os_getenv.return_value = "fake_api_key" # Mock API key

        mock_client_instance = MagicMock()
        mock_google_genai_Client.return_value = mock_client_instance

        mock_response = MagicMock()
        expected_response_text = '{"key": "value"}'
        mock_response.text = expected_response_text
        mock_client_instance.models.generate_content.return_value = mock_response

        controller = GeminiController(model="gemini-pro", api_key="fake_api_key")

        prompt = "Test prompt for JSON"
        response_format = {
            "type": "json_object",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}}
                }
            }
        }
        sample_schema = response_format["json_schema"]["schema"]

        # Act
        actual_response = controller.get_completion(prompt, response_format=response_format, temperature=0.5)

        # Assert
        self.assertEqual(actual_response, expected_response_text)
        mock_client_instance.models.generate_content.assert_called_once()
        call_args = mock_client_instance.models.generate_content.call_args

        self.assertEqual(call_args.kwargs['model'], controller.model_name)
        self.assertEqual(call_args.kwargs['contents'], prompt)

        generation_config = call_args.kwargs['generation_config']
        self.assertIsInstance(generation_config, genai.types.GenerationConfig) # Corrected type
        self.assertEqual(generation_config.temperature, 0.5)
        self.assertEqual(generation_config.response_mime_type, "application/json")
        self.assertEqual(generation_config.response_schema, sample_schema)

    @patch('os.getenv')
    @patch('google.generativeai.Client') # Corrected patch target
    def test_gemini_get_completion_text_output(self, mock_google_generativeai_Client, mock_os_getenv):
        # Arrange
        mock_os_getenv.return_value = "fake_api_key"

        mock_client_instance = MagicMock()
        mock_google_genai_Client.return_value = mock_client_instance

        mock_response = MagicMock()
        expected_response_text = "This is a text response."
        mock_response.text = expected_response_text
        mock_client_instance.models.generate_content.return_value = mock_response

        controller = GeminiController(model="gemini-pro", api_key="fake_api_key")

        prompt = "Test prompt for text"

        # Act
        actual_response = controller.get_completion(prompt, temperature=0.6)

        # Assert
        self.assertEqual(actual_response, expected_response_text)
        mock_client_instance.models.generate_content.assert_called_once()
        call_args = mock_client_instance.models.generate_content.call_args

        self.assertEqual(call_args.kwargs['model'], controller.model_name)
        self.assertEqual(call_args.kwargs['contents'], prompt)

        generation_config = call_args.kwargs['generation_config']
        self.assertIsInstance(generation_config, genai.types.GenerationConfig) # Corrected type
        self.assertEqual(generation_config.temperature, 0.6)
        self.assertIsNone(getattr(generation_config, 'response_mime_type', None)) # Should not be set
        self.assertIsNone(getattr(generation_config, 'response_schema', None)) # Should not be set

    @patch('os.getenv')
    @patch('google.generativeai.Client') # Corrected patch target
    def test_gemini_get_completion_api_error(self, mock_google_generativeai_Client, mock_os_getenv):
        # Arrange
        mock_os_getenv.return_value = "fake_api_key"

        mock_client_instance = MagicMock()
        mock_google_generativeai_Client.return_value = mock_client_instance

        # Configure generate_content to raise an APIError
        mock_client_instance.models.generate_content.side_effect = genai.errors.APIError("Test API Error") # Corrected error type

        controller = GeminiController(model="gemini-pro", api_key="fake_api_key")

        prompt = "Test prompt that causes error"
        response_format_json = {"type": "json_object"}

        # Act & Assert for JSON response format
        json_error_response = controller.get_completion(prompt, response_format=response_format_json)
        self.assertEqual(json_error_response, "{}", "Should return empty JSON string on API error for JSON requests")

        # Reset mock for next call if necessary (side_effect persists)
        mock_client_instance.models.generate_content.side_effect = genai.errors.APIError("Test API Error") # Corrected error type

        # Act & Assert for text response format
        text_error_response = controller.get_completion(prompt) # No specific format, expects text
        self.assertTrue("Error generating content with Gemini: Test API Error" in text_error_response,
                        "Should return error message on API error for text requests")

        # Check generate_content was called (twice in this test case)
        self.assertEqual(mock_client_instance.models.generate_content.call_count, 2)

    @patch('os.getenv')
    @patch('google.generativeai.Client') # Corrected patch target
    def test_gemini_get_completion_blocked_prompt_error_json(self, mock_google_generativeai_Client, mock_os_getenv):
        # Arrange
        mock_os_getenv.return_value = "fake_api_key"
        mock_client_instance = MagicMock()
        mock_google_generativeai_Client.return_value = mock_client_instance

        mock_client_instance.models.generate_content.side_effect = genai.errors.BlockedPromptException("Prompt blocked") # Corrected error type
        controller = GeminiController(api_key="fake_api_key")
        prompt = "A problematic prompt"
        response_format = {"type": "json_object"}

        # Act
        response = controller.get_completion(prompt, response_format=response_format)

        # Assert
        self.assertEqual(response, "{}")
        mock_client_instance.models.generate_content.assert_called_once()


    @patch('os.getenv')
    @patch('google.generativeai.Client') # Corrected patch target
    def test_gemini_get_completion_empty_response_value_error_json(self, mock_google_generativeai_Client, mock_os_getenv):
        # Arrange
        mock_os_getenv.return_value = "fake_api_key"
        mock_client_instance = MagicMock()
        mock_google_generativeai_Client.return_value = mock_client_instance

        # Simulate a response object that would lead to the specific ValueError for empty/problematic content
        mock_empty_response = MagicMock()
        mock_empty_response.parts = [] # No parts
        mock_empty_response.prompt_feedback = None # No immediate block reason in feedback
        # If the internal logic relies on candidates for finish_reason:
        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "OTHER" # Or any reason that's not explicitly handled to show blockage
        mock_empty_response.candidates = [mock_candidate]

        mock_client_instance.models.generate_content.return_value = mock_empty_response

        controller = GeminiController(api_key="fake_api_key")
        prompt = "A prompt leading to empty response"
        response_format = {"type": "json_object"}

        # Act
        response = controller.get_completion(prompt, response_format=response_format)

        # Assert
        # Based on current llm_controller.py, an empty response without explicit block
        # but leading to ValueError (e.g. "Gemini API returned an empty response. Finish reason: OTHER")
        # will be caught by the generic ValueError handler.
        self.assertEqual(response, "{}")
        mock_client_instance.models.generate_content.assert_called_once()


class TestLLMControllerLogging(unittest.TestCase):

    def test_get_completion_with_stage_calls_logger(self):
        mock_backend_llm = MagicMock()
        mock_backend_llm.get_completion.return_value = "mocked response"

        mock_logger = MagicMock(spec=LLMInteractionLogger)

        # Patch the specific backend controller that LLMController would instantiate.
        # Assuming default backend is 'openai' or it's explicitly set.
        with patch('llm_controller.OpenAIController', return_value=mock_backend_llm) as MockOpenAI:
            # Ensure LLMController uses the 'openai' backend string to trigger MockOpenAI
            controller = LLMController(backend='openai', logger=mock_logger)

            test_prompt = "Hello LLM"
            test_stage = "TestStage"

            response = controller.get_completion(prompt=test_prompt, stage=test_stage)

            self.assertEqual(response, "mocked response")
            # Check that the *mocked backend's* get_completion was called by LLMController
            mock_backend_llm.get_completion.assert_called_once_with(test_prompt, response_format=None, temperature=0.7)
            # Check that the logger was called with the correct parameters
            mock_logger.log.assert_called_once_with(stage=test_stage, prompt=test_prompt, response="mocked response")

    def test_get_completion_without_stage_does_not_call_logger(self):
        mock_backend_llm = MagicMock()
        mock_backend_llm.get_completion.return_value = "mocked response"
        mock_logger = MagicMock(spec=LLMInteractionLogger)

        with patch('llm_controller.OpenAIController', return_value=mock_backend_llm):
            controller = LLMController(backend='openai', logger=mock_logger)
            test_prompt = "Hello again"
            response = controller.get_completion(prompt=test_prompt) # No stage

            self.assertEqual(response, "mocked response")
            mock_backend_llm.get_completion.assert_called_once_with(test_prompt, response_format=None, temperature=0.7)
            mock_logger.log.assert_not_called()

    def test_get_completion_with_stage_but_no_logger(self):
        mock_backend_llm = MagicMock()
        mock_backend_llm.get_completion.return_value = "mocked response"

        with patch('llm_controller.OpenAIController', return_value=mock_backend_llm):
            controller = LLMController(backend='openai', logger=None) # No logger
            test_prompt = "Hello one more time"
            test_stage = "TestStageNoLogger"

            try:
                response = controller.get_completion(prompt=test_prompt, stage=test_stage)
                self.assertEqual(response, "mocked response")
            except Exception as e:
                self.fail(f"LLMController raised an exception when no logger was provided: {e}")

            mock_backend_llm.get_completion.assert_called_once_with(test_prompt, response_format=None, temperature=0.7)
            # No specific logger call to assert as it shouldn't exist or be called.
            # The main check is that no error occurred.

if __name__ == '__main__':
    unittest.main()
