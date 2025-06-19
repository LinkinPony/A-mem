import logging
import configparser
from typing import Optional # Import Optional
from datetime import datetime
import os
import sys # Import sys for sys.stdout

class LLMInteractionLogger:
    def __init__(self,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 log_file_path: str = "llm_interactions.log"):
        self.logger = logging.getLogger(__name__ + ".LLMInteractionLogger")
        # Initialize attributes that will be set by _configure_logger
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self._configure_logger()

    def _configure_logger(self):
        config = configparser.ConfigParser()
        config_path = "config.ini"

        # Start with constructor/default values
        current_log_to_console = self.log_to_console
        current_log_to_file = self.log_to_file
        current_log_file_path = self.log_file_path
        log_level_str = "INFO" # Default log level

        if os.path.exists(config_path):
            config.read(config_path)
            if "logging" in config:
                log_level_str = config.get("logging", "log_level", fallback="INFO").upper()
                current_log_to_console = config.getboolean("logging", "log_to_console", fallback=self.log_to_console)
                current_log_to_file = config.getboolean("logging", "log_to_file", fallback=self.log_to_file)
                current_log_file_path = config.get("logging", "log_file_path", fallback=self.log_file_path)

        # Apply the determined settings
        self.log_to_console = current_log_to_console
        self.log_to_file = current_log_to_file
        self.log_file_path = current_log_file_path

        log_level = getattr(logging, log_level_str, logging.INFO)
        self.logger.setLevel(log_level)

        # Close and remove existing handlers
        for handler in self.logger.handlers[:]: # Iterate over a copy
            handler.close()
            self.logger.removeHandler(handler)

        formatter = logging.Formatter('%(message)s') # We format the message in the log method

        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout) # Explicitly use sys.stdout
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.log_to_file:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError as e:
                    self.logger.error(f"Could not create log directory {log_dir}: {e}", exc_info=True)
                    self.log_to_file = False # Disable file logging if dir creation fails

            if self.log_to_file: # Check again if disabled by directory creation failure
                try:
                    file_handler = logging.FileHandler(self.log_file_path, mode='a', encoding='utf-8')
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except (IOError, OSError) as e:
                    self.logger.error(f"Could not set up file handler for {self.log_file_path}: {e}", exc_info=True)
                    self.log_to_file = False # Disable if handler setup fails

        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def log(self, stage: str, prompt: str, response: Optional[str] = None):
        # Check if the logger is configured to handle messages at its set level
        # This check is implicitly handled by logger.log() if its level is high enough.
        # Explicit check: if not self.logger.isEnabledFor(self.logger.level): return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if response is None:
            # Pre-request logging
            log_message = f'''
============================================================
[LLM Interaction Log - PRE-REQUEST]
Stage    : {stage}
Timestamp: {timestamp}
============================================================

[PROMPT]
------------------------------------------------------------
{prompt}
------------------------------------------------------------
'''
        else:
            # Post-response logging
            log_message = f'''
============================================================
[LLM Interaction Log - POST-RESPONSE]
Stage    : {stage}
Timestamp: {timestamp}
============================================================

[PROMPT]
------------------------------------------------------------
{prompt}
------------------------------------------------------------

[RESPONSE]
------------------------------------------------------------
{response}
------------------------------------------------------------
'''
        # Log with the logger's effective level.
        # self.logger.level is the threshold. We can use a specific level like INFO or DEBUG
        # For this purpose, logging at the logger's own level is appropriate.
        self.logger.log(self.logger.level, log_message.strip())
