import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Literal, Any

# 为 Gemini 更新了导入
import google.generativeai as genai # Corrected import
# types and errors will be accessed via genai.types and genai.errors
from litellm import completion

from llm_interaction_logger import LLMInteractionLogger # Added import


class BaseLLMController(ABC):
    """所有 LLM 控制器的抽象基类。"""

    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """从 LLM 获取补全。"""


class OpenAIController(BaseLLMController):
    """用于与 OpenAI API 交互的控制器。"""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError("OpenAI package not found. Install it with: pip install openai") from exc

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """使用 OpenAI 的 Chat Completions API 获取补全。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[  # type: ignore
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,  # type: ignore
            temperature=temperature,
            max_tokens=1000
        )
        # response.choices[0].message.content 在新版 openai-python 中可能为 None
        return response.choices[0].message.content or ""


class OllamaController(BaseLLMController):
    """用于与 Ollama 交互的控制器。"""

    def __init__(self, model: str = "llama2"):
        self.model = model

    def _generate_empty_value(self, schema_type: str) -> Any:
        """根据 JSON schema 类型生成一个空的默认值。"""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """当 API 调用失败时，根据 schema 生成一个空的 JSON 响应。"""
        if not response_format or "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"].get("schema", {})
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema.get("type", ""))

        return result

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """使用 litellm 与 Ollama 模型进行交互。"""
        try:
            # litellm 似乎不直接使用 temperature 参数，但我们保留它以保持接口一致
            response = completion(
                model=f"ollama_chat/{self.model}",
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content or ""
        # pylint: disable=broad-exception-caught
        except Exception:
            # 如果 litellm/Ollama 调用失败，返回一个基于 schema 的空 JSON 字符串
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)


class GeminiController(BaseLLMController):
    """用于与 Google Gen AI SDK 交互的控制器（根据新版 SDK 更新）。"""

    def __init__(self, model: str = "gemini-pro", api_key: Optional[str] = None):
        self.model_name = model
        try:
            # 优先使用传入的 api_key，否则从环境变量查找
            if api_key is None:
                # 新 SDK 文档推荐使用 GOOGLE_API_KEY
                api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if api_key is None:
                raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

            # 根据新 SDK 文档，使用 genai.Client 初始化
            self.client = genai.Client(api_key=api_key)

        except ImportError as exc:
            raise ImportError(
                "Google Gen AI SDK not found. Install it with: pip install google-genai") from exc
        except Exception as e:
            raise ValueError(f"Error initializing Gemini client: {e}") from e

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """使用 google-genai SDK 获取补全。"""
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature

        # 为 JSON 输出模式配置正确的参数
        if response_format and response_format.get("type") in ["json_schema", "json_object"]:
            config_params["response_mime_type"] = "application/json"
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                config_params["response_schema"] = schema

        # 根据新 SDK 文档，使用 GenerateContentConfig
        config = genai.types.GenerateContentConfig(**config_params) # Corrected access to types

        try:
            # 根据新 SDK 文档，使用 client.models.generate_content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text

        # 根据新 SDK 文档，使用 errors.APIError
        except (genai.errors.APIError, ValueError) as e: # Corrected access to errors
            if response_format and response_format.get("type") in ["json_schema", "json_object"]:
                return "{}"  # 对于 JSON 请求，在出错时返回空对象
            return f"Error generating content with Gemini: {str(e)}"
        # pylint: disable=broad-exception-caught
        except Exception as e:
            if response_format and response_format.get("type") in ["json_schema", "json_object"]:
                return "{}"
            return f"Unexpected error with Gemini: {str(e)}"


class LLMController:
    """LLM 控制器，用于生成元数据。"""

    def __init__(self,
                 backend: Literal["openai", "ollama", "gemini"] = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 logger: Optional[LLMInteractionLogger] = None): # Added logger parameter
        self.backend = backend
        self.logger = logger # Added logger instance variable
        if backend == "openai":
            self.llm: BaseLLMController = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "gemini":
            self.llm = GeminiController(model, api_key)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'gemini'")

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7, stage: Optional[str] = None) -> str: # Added stage parameter
        """根据所选后端获取 LLM 补全。"""
        # The actual call to the specific LLM backend (OpenAI, Ollama, Gemini)
        # should NOT pass the 'stage' parameter. Their get_completion methods remain unchanged.
        response = self.llm.get_completion(prompt, response_format=response_format, temperature=temperature)

        if self.logger and stage:
            self.logger.log(stage=stage, prompt=prompt, response=response)

        return response
