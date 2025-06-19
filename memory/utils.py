import json

# Helper function to reliably extract JSON from LLM responses.
# This prevents errors when the LLM wraps JSON in markdown code blocks.
def _extract_json_from_response(response_text: str) -> str:
    """
    Extracts a JSON string from a raw LLM response that might be wrapped
    in markdown code blocks (e.g., ```json ... ```).
    """
    if not isinstance(response_text, str):
        return ""

    # Find the first occurrence of '{' which often marks the beginning of the JSON object
    start_brace = response_text.find('{')
    if start_brace == -1:
        return ""  # No JSON object found

    # Find the last occurrence of '}' which often marks the end of the JSON object
    end_brace = response_text.rfind('}')
    if end_brace == -1:
        return ""  # No JSON object found

    # Extract the substring from the first '{' to the last '}'
    json_str = response_text[start_brace: end_brace + 1]

    return json_str
