"""
Model-agnostic client interfaces for PromptInjector.
Supports any AI model or API endpoint through generic HTTP calls.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp


@dataclass
class ModelResponse:
    """Generic response from any AI model"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class ModelRequest:
    """Generic request to any AI model"""
    prompt: str
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    system_prompt: Optional[str] = None


class BaseModelClient(ABC):
    """Abstract base class for AI model clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from the AI model"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the model endpoint is accessible"""
        pass


class HTTPModelClient(BaseModelClient):
    """Generic HTTP client for any AI model API"""
    
    def __init__(self, 
                 endpoint_url: str,
                 api_key: str,
                 model: str = "default",
                 headers: Optional[Dict[str, str]] = None,
                 request_formatter: Optional[callable] = None,
                 response_parser: Optional[callable] = None,
                 **config):
        super().__init__(config)
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.headers = headers or {}
        self.request_formatter = request_formatter or self._default_request_formatter
        self.response_parser = response_parser or self._default_response_parser
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # Default headers
        if 'Authorization' not in self.headers and api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'
    
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using generic HTTP POST"""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Format request payload
                payload = self.request_formatter(request)
                
                # Log outgoing request to target LLM
                self.logger.info(f"=== REQUEST TO TARGET LLM ===")
                self.logger.info(f"Endpoint: {self.endpoint_url}")
                self.logger.info(f"Model: {request.model}")
                self.logger.info(f"Prompt: {request.prompt[:500]}{'...' if len(request.prompt) > 500 else ''}")
                self.logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
                
                # Make HTTP request
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        self.endpoint_url,
                        json=payload,
                        headers=self.headers
                    ) as response:
                        
                        if response.status == 200:
                            response_data = await response.json()
                            parsed_response = self.response_parser(response_data, request.model)
                            
                            # Log response from target LLM
                            self.logger.info(f"=== RESPONSE FROM TARGET LLM ===")
                            self.logger.info(f"Status: {response.status}")
                            self.logger.info(f"Model: {request.model}")
                            self.logger.info(f"Response content: {parsed_response.content[:500]}{'...' if len(parsed_response.content) > 500 else ''}")
                            if parsed_response.usage:
                                self.logger.info(f"Token usage: {parsed_response.usage}")
                            
                            return parsed_response
                        else:
                            error_text = await response.text()
                            self.logger.error(f"=== TARGET LLM ERROR ===")
                            self.logger.error(f"Status: {response.status}")
                            self.logger.error(f"Error: {error_text}")
                            raise Exception(f"HTTP {response.status}: {error_text}")
                            
            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    async def test_connection(self) -> bool:
        """Test connection to the model endpoint"""
        try:
            test_request = ModelRequest(
                prompt="Hello",
                model=self.model,
                max_tokens=10
            )
            await self.generate_response(test_request)
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def _default_request_formatter(self, request: ModelRequest) -> Dict[str, Any]:
        """Default request formatter - can be overridden for specific APIs"""
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        
        if request.system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": request.system_prompt})
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
            
        return payload
    
    def _default_response_parser(self, response_data: Dict[str, Any], model: str) -> ModelResponse:
        """Default response parser - can be overridden for specific APIs"""
        # Try common response formats
        
        # OpenAI-style format
        if "choices" in response_data:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            return ModelResponse(
                content=content,
                usage=usage,
                model=model,
                metadata={"raw_response": response_data}
            )
        
        # Simple text response
        if "response" in response_data:
            return ModelResponse(
                content=response_data["response"],
                model=model,
                metadata={"raw_response": response_data}
            )
        
        # Direct text
        if isinstance(response_data, str):
            return ModelResponse(
                content=response_data,
                model=model
            )
        
        # Fallback - look for any text field
        for key in ["text", "output", "result", "generated_text"]:
            if key in response_data:
                return ModelResponse(
                    content=response_data[key],
                    model=model,
                    metadata={"raw_response": response_data}
                )
        
        raise ValueError(f"Unable to parse response format: {response_data}")


class OpenAIClient(HTTPModelClient):
    """OpenAI-specific client implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", endpoint_url: str = "https://api.openai.com/v1/chat/completions", **config):
        # Remove conflicting parameters from config
        clean_config = {k: v for k, v in config.items() if k not in ['endpoint_url', 'api_key', 'model']}
        super().__init__(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model=model,
            **clean_config
        )


class OllamaClient(HTTPModelClient):
    """Ollama local model client"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", **config):
        # Remove conflicting parameters from config
        clean_config = {k: v for k, v in config.items() if k not in ['endpoint_url', 'api_key']}
        super().__init__(
            endpoint_url=f"{base_url}/api/generate",
            api_key="",  # Ollama doesn't require API key
            **clean_config
        )
        self.model = model
    
    def _default_request_formatter(self, request: ModelRequest) -> Dict[str, Any]:
        """Ollama-specific request format"""
        return {
            "model": request.model or self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens or 500,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
            }
        }
    
    def _default_response_parser(self, response_data: Dict[str, Any], model: str) -> ModelResponse:
        """Ollama-specific response parser"""
        return ModelResponse(
            content=response_data.get("response", ""),
            model=model,
            metadata={"raw_response": response_data}
        )


class AnthropicClient(HTTPModelClient):
    """Anthropic Claude client"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1/messages", **config):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        # Remove conflicting parameters from config
        clean_config = {k: v for k, v in config.items() if k not in ['endpoint_url', 'api_key', 'headers']}
        super().__init__(
            endpoint_url=base_url,
            api_key="",  # API key is in headers
            headers=headers,
            **clean_config
        )
    
    def _default_request_formatter(self, request: ModelRequest) -> Dict[str, Any]:
        """Anthropic-specific request format"""
        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens or 500,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
            
        return payload
    
    def _default_response_parser(self, response_data: Dict[str, Any], model: str) -> ModelResponse:
        """Anthropic-specific response parser"""
        content = response_data["content"][0]["text"]
        return ModelResponse(
            content=content,
            usage=response_data.get("usage"),
            model=model,
            metadata={"raw_response": response_data}
        )


class ModelClientFactory:
    """Factory for creating model clients"""
    
    @staticmethod
    def create_client(client_type: str, **config) -> BaseModelClient:
        """Create a model client based on type"""
        
        client_type = client_type.lower()
        
        if client_type == "openai":
            return OpenAIClient(**config)
        elif client_type == "ollama":
            return OllamaClient(**config)
        elif client_type == "anthropic":
            return AnthropicClient(**config)
        elif client_type == "http" or client_type == "generic":
            return HTTPModelClient(**config)
        else:
            raise ValueError(f"Unknown client type: {client_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseModelClient:
        """Create client from configuration dictionary"""
        client_type = config.get("type", "http")
        return ModelClientFactory.create_client(client_type, **config)


# Convenience function for backward compatibility
async def test_model_client(client: BaseModelClient, test_prompt: str = "Hello, how are you?") -> bool:
    """Test a model client with a simple prompt"""
    try:
        request = ModelRequest(
            prompt=test_prompt,
            model=client.config.get('model', 'default'),
            max_tokens=50
        )
        response = await client.generate_response(request)
        print(f"Test successful! Response: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    async def demo():
        # Test OpenAI client
        print("Testing OpenAI client...")
        openai_client = OpenAIClient(
            api_key="your-api-key-here",
            model="gpt-3.5-turbo"
        )
        
        # Test HTTP client with custom endpoint
        print("Testing generic HTTP client...")
        custom_client = HTTPModelClient(
            endpoint_url="https://api.example.com/v1/chat",
            api_key="your-api-key",
            model="custom-model"
        )
        
        # Test Ollama client
        print("Testing Ollama client...")
        ollama_client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama2"
        )
        
        # You can test any of these clients:
        # await test_model_client(openai_client)
        # await test_model_client(custom_client)
        # await test_model_client(ollama_client)
    
    # asyncio.run(demo())
    print("Model client module loaded successfully!")