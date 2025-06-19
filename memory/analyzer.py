import json
import logging
from typing import Dict

from llm_controller import LLMController
from memory.utils import _extract_json_from_response

logger = logging.getLogger(__name__)

class Analyzer:
    def __init__(self, llm_controller: LLMController):
        self.llm_controller = llm_controller

    def analyze_content(self, content: str) -> Dict:
        """Analyze content using LLM to extract semantic metadata.

        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories

        Args:
            content (str): The text content to analyze

        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context":
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.get_completion(prompt, # Corrected: call get_completion on llm_controller directly
                                                              response_format={"type": "json_schema", "json_schema": {
                                                                  "name": "response",
                                                                  "schema": {
                                                                      "type": "object",
                                                                      "properties": {
                                                                          "keywords": {
                                                                              "type": "array",
                                                                              "items": {
                                                                                  "type": "string"
                                                                              }
                                                                          },
                                                                          "context": {
                                                                              "type": "string",
                                                                          },
                                                                          "tags": {
                                                                              "type": "array",
                                                                              "items": {
                                                                                  "type": "string"
                                                                              }
                                                                          }
                                                                      }
                                                                  }
                                                              }},
                                                              stage="Content Analysis") # Added stage parameter
            # Clean the response before parsing to handle markdown code blocks.
            clean_json_str = _extract_json_from_response(response)
            if not clean_json_str:
                raise json.JSONDecodeError("Failed to extract JSON from analyze_content response.", response, 0)
            return json.loads(clean_json_str)
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}
