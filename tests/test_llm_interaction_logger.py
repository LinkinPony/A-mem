import unittest
import os
import configparser
import logging
from io import StringIO
from unittest.mock import patch, mock_open

# Adjust import path based on your project structure
# If llm_interaction_logger.py is in the root directory:
from llm_interaction_logger import LLMInteractionLogger

TEST_CONFIG_FILE = "test_config.ini"
TEST_LOG_FILE = "test_llm_interactions.log"

class TestLLMInteractionLogger(unittest.TestCase):

    def setUp(self):
        # Ensure a clean state before each test
        self._remove_test_files()
        # Create a default test config
        self.config = configparser.ConfigParser()
        self.config['logging'] = {
            'log_level': 'INFO',
            'log_to_console': 'True',
            'log_to_file': 'True',
            'log_file_path': TEST_LOG_FILE
        }
        with open(TEST_CONFIG_FILE, 'w') as f:
            self.config.write(f)

        # It's important that LLMInteractionLogger reads "config.ini", not "test_config.ini" directly
        # So we will rename our test_config.ini to config.ini before logger instantiation
        if os.path.exists("config.ini"):
            # If config.ini.bak exists, it means a previous test run might have failed
            # during or after renaming config.ini.bak to config.ini, but before cleaning up config.ini.bak
            # Or, config.ini is a leftover from a failed test that should have been cleaned.
            # Prioritize restoring backup if it exists.
            if os.path.exists("config.ini.bak"):
                os.remove("config.ini") # Remove current config.ini to allow rename from .bak
            else:
                # If no backup, current config.ini might be a leftover or the one we are about to create over.
                # It's safer to rename it to .bak before creating the new one.
                os.rename("config.ini", "config.ini.bak")
        os.rename(TEST_CONFIG_FILE, "config.ini")


    def tearDown(self):
        self._remove_test_files()
        # Restore original config.ini if it was backed up
        if os.path.exists("config.ini.bak"):
            if os.path.exists("config.ini"): # if config.ini was created by logger (e.g. if test failed before cleanup)
                 os.remove("config.ini")
            os.rename("config.ini.bak", "config.ini")
        elif os.path.exists("config.ini"): # if no backup, but test created config.ini
            # This case implies the logger used the test config renamed to config.ini
            # and tearDown is responsible for cleaning it up.
            # Check if it's our test config before removing
            cfg_check = configparser.ConfigParser()
            cfg_check.read("config.ini")
            if cfg_check.get("logging", "log_file_path", fallback="") == TEST_LOG_FILE:
                os.remove("config.ini")


    def _remove_test_files(self):
        if os.path.exists(TEST_LOG_FILE):
            os.remove(TEST_LOG_FILE)
        if os.path.exists(TEST_CONFIG_FILE): # This is the one named test_config.ini
            os.remove(TEST_CONFIG_FILE)
        # config.ini is handled by rename in setup/teardown if necessary

    def test_log_formatting_and_output(self):
        stage = "TestStage"
        prompt = "Test prompt message."
        response = "Test response message."

        # Test console output
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Instantiate logger AFTER stdout is patched
            logger = LLMInteractionLogger() # Reads from "config.ini" (which is our TEST_CONFIG_FILE)
            logger.log(stage, prompt, response)
            output = fake_stdout.getvalue()
            self.assertIn("[LLM Interaction Log]", output)
            self.assertIn(f"Stage    : {stage}", output)
            self.assertIn(f"[PROMPT]", output)
            self.assertIn(prompt, output)
            self.assertIn(f"[RESPONSE]", output)
            self.assertIn(response, output)
            # Ensure it's the post-response header
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", output)
            self.assertNotIn("[LLM Interaction Log - PRE-REQUEST]", output)


        # Test file output
        self.assertTrue(os.path.exists(TEST_LOG_FILE))
        with open(TEST_LOG_FILE, 'r') as f:
            content = f.read()
            # Check pre-request log in file
            self.assertIn("[LLM Interaction Log - PRE-REQUEST]", content)
            self.assertRegex(content, rf"\[LLM Interaction Log - PRE-REQUEST\].*Stage\s*:\s*{stage}.*\[PROMPT\].*{prompt}", os.DOTALL)
            # Check that response section is NOT in the pre-request part of the log specifically for this stage
            # This is tricky with full file content. We rely on the console check's specificity.
            # A simpler check for the file is that the prompt exists for pre-request
            # and then check the post-request part separately.

            # Check post-request log in file
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", content)
            self.assertRegex(content, rf"\[LLM Interaction Log - POST-RESPONSE\].*Stage\s*:\s*{stage}.*\[PROMPT\].*{prompt}.*\[RESPONSE\].*{response}", os.DOTALL)

            # Verify ordering if possible, or at least presence of both distinct logs
            pre_log_index = content.find("[LLM Interaction Log - PRE-REQUEST]")
            post_log_index = content.find("[LLM Interaction Log - POST-RESPONSE]")
            self.assertTrue(pre_log_index != -1 and post_log_index != -1, "Both pre and post logs should be in the file")
            # This doesn't strictly guarantee the response isn't in the pre-request block for the same stage,
            # but combined with console checks, it's reasonably robust.

    def test_log_formatting_pre_request_only(self):
        stage = "PreRequestStage"
        prompt = "Pre-request prompt only."

        # Test console output for pre-request
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            logger = LLMInteractionLogger()
            logger.log(stage, prompt) # Log without response
            output = fake_stdout.getvalue()

            self.assertIn("[LLM Interaction Log - PRE-REQUEST]", output)
            self.assertIn(f"Stage    : {stage}", output)
            self.assertIn(f"[PROMPT]", output)
            self.assertIn(prompt, output)
            self.assertNotIn("[RESPONSE]", output) # Crucial for pre-request

        # Test file output for pre-request
        self.assertTrue(os.path.exists(TEST_LOG_FILE))
        with open(TEST_LOG_FILE, 'r') as f:
            content = f.read()
            self.assertIn("[LLM Interaction Log - PRE-REQUEST]", content)
            self.assertIn(f"Stage    : {stage}", content)
            self.assertIn(prompt, content)
            self.assertNotIn("[RESPONSE]", content) # Crucial for pre-request

    def test_error_string_logging(self):
        stage = "ErrorStage"
        prompt = "Prompt that led to an error."
        error_response = "API Error: Something went terribly wrong."

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            logger = LLMInteractionLogger()
            logger.log(stage, prompt, error_response) # Log with error string as response
            output = fake_stdout.getvalue()

            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", output)
            self.assertIn(f"Stage    : {stage}", output)
            self.assertIn(f"[PROMPT]", output)
            self.assertIn(prompt, output)
            self.assertIn(f"[RESPONSE]", output)
            self.assertIn(error_response, output) # Check if the error message is logged

        self.assertTrue(os.path.exists(TEST_LOG_FILE))
        with open(TEST_LOG_FILE, 'r') as f:
            content = f.read()
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", content)
            self.assertIn(error_response, content)

    def test_log_level_debug(self):
        # Modify config for this test
        self.config['logging']['log_level'] = 'DEBUG'
        with open("config.ini", 'w') as f: # Overwrite the "config.ini" (which was TEST_CONFIG_FILE)
            self.config.write(f)

        # Instantiate logger here to check its level attribute before patching stdout
        logger_for_level_check = LLMInteractionLogger()
        self.assertEqual(logger_for_level_check.logger.level, logging.DEBUG)

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Instantiate logger again inside patch for console output testing
            logger = LLMInteractionLogger()
            # Log full message to ensure it's processed
            logger.log("DebugStage", "Debug prompt", "Debug response")
            output = fake_stdout.getvalue()
            self.assertIn("DebugStage", output)
            # Check for new header format
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", output)


        # File check should be fine with any logger instance that used the config
        with open(TEST_LOG_FILE, 'r') as f:
            content = f.read()
            self.assertIn("DebugStage", content)
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", content)


    def test_log_level_warning_suppresses_info(self):
        self.config['logging']['log_level'] = 'WARNING'
        with open("config.ini", 'w') as f:
            self.config.write(f)

        # Instantiate logger here to check its level attribute before patching stdout
        logger_for_level_check = LLMInteractionLogger()
        self.assertEqual(logger_for_level_check.logger.level, logging.WARNING)

        # This log call uses logger.logger.level which is WARNING.
        # The message itself is not INFO or DEBUG, it's just a message.
        # The logger.log method logs at logger.level. So it should appear.
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Instantiate logger again inside patch for console output testing
            logger = LLMInteractionLogger()
            # Log full message
            logger.log("InfoStage", "Info prompt", "Info response")
            output = fake_stdout.getvalue()
            # If logger level is WARNING, and we log at WARNING (which is logger.level), it should appear.
            self.assertIn("InfoStage", output)
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", output)

        # This test, as originally written and still, confirms that messages logged
        # via LLMInteractionLogger.log() are emitted if the logger's level is
        # appropriate. It doesn't test suppression of lower-level direct calls
        # (e.g., logger.logger.info()) because LLMInteractionLogger.log() is the
        # sole public logging method for interactions.

    def test_log_to_console_disabled(self):
        self.config['logging']['log_to_console'] = 'False'
        with open("config.ini", 'w') as f:
            self.config.write(f)

        logger = LLMInteractionLogger()
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            logger.log("ConsoleTest", "Prompt", "Response")
            self.assertEqual(fake_stdout.getvalue(), "") # Should be empty

    def test_log_to_file_disabled(self):
        self._remove_test_files() # Ensure log file doesn't exist from previous interactions
        self.config['logging']['log_to_file'] = 'False'
        # We still need a path for the config, even if not used.
        self.config['logging']['log_file_path'] = TEST_LOG_FILE
        with open("config.ini", 'w') as f:
            self.config.write(f)

        logger = LLMInteractionLogger() # Reads config
        logger.log("FileTest", "Prompt", "Response")
        self.assertFalse(os.path.exists(TEST_LOG_FILE))

    def test_default_config_values_if_config_missing(self):
        # Remove config.ini entirely
        if os.path.exists("config.ini"):
            os.remove("config.ini")

        # For checking attributes like log_to_console, log_to_file, level, path
        # we can instantiate it once outside the patch.
        logger_for_attr_check = LLMInteractionLogger(log_file_path=TEST_LOG_FILE)
        self.assertTrue(logger_for_attr_check.log_to_console) # Default is True
        self.assertTrue(logger_for_attr_check.log_to_file)   # Default is True
        self.assertEqual(logger_for_attr_check.logger.level, logging.INFO) # Default is INFO
        self.assertEqual(logger_for_attr_check.log_file_path, TEST_LOG_FILE)

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Instantiate new logger for console check under patch
            logger_for_console_check = LLMInteractionLogger(log_file_path=TEST_LOG_FILE)
            # Log full message
            logger_for_console_check.log("DefaultTest", "Prompt", "Response")
            output = fake_stdout.getvalue()
            self.assertIn("DefaultTest", output)
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", output)


        # File check can use the first instance or a new one, should reflect the log call
        self.assertTrue(os.path.exists(TEST_LOG_FILE))
        with open(TEST_LOG_FILE, 'r') as f:
            content = f.read()
            self.assertIn("DefaultTest", content)
            self.assertIn("[LLM Interaction Log - POST-RESPONSE]", content)


    def test_log_directory_creation(self):
        nested_log_file = os.path.join("test_logs_dir", "nested_log.log")
        log_dir = os.path.dirname(nested_log_file)
        if os.path.exists(log_dir):
            # Clean up from previous failed test if necessary
            for f in os.listdir(log_dir): os.remove(os.path.join(log_dir, f))
            os.rmdir(log_dir)

        self.config['logging']['log_file_path'] = nested_log_file
        with open("config.ini", 'w') as f:
            self.config.write(f)

        logger = LLMInteractionLogger()
        logger.log("NestedDirTest", "Prompt", "Response")

        self.assertTrue(os.path.exists(log_dir))
        self.assertTrue(os.path.exists(nested_log_file))

        # Cleanup
        if os.path.exists(nested_log_file):
            os.remove(nested_log_file)
        if os.path.exists(log_dir):
            os.rmdir(log_dir)

if __name__ == '__main__':
    unittest.main()
