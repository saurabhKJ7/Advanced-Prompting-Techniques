"""
LLM Interface Module

This module provides a unified interface for interacting with different
Large Language Model APIs, supporting OpenAI, Anthropic, and other providers.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from enum import Enum

import openai
import anthropic
import httpx
from pydantic import BaseModel, Field


class ModelProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Standard response format for all LLM interactions."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]
    timestamp: float
    latency: float
    error: Optional[str] = None


@dataclass
class LLMRequest:
    """Standard request format for all LLM interactions."""
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    def __init__(self, api_key: str, model: str, default_config: Dict[str, Any] = None):
        self.api_key = api_key
        self.model = model
        self.default_config = default_config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "error_count": 0
        }
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests."""
        pass
    
    @abstractmethod
    def get_provider(self) -> ModelProvider:
        """Return the provider enum."""
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validate if the model is supported by this provider."""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this interface."""
        return self._usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "error_count": 0
        }
    
    def _update_usage_stats(self, response: LLMResponse):
        """Update usage statistics based on response."""
        self._usage_stats["total_requests"] += 1
        if response.usage:
            self._usage_stats["total_tokens"] += response.usage.get("total_tokens", 0)
        if response.error:
            self._usage_stats["error_count"] += 1


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI GPT models."""
    
    SUPPORTED_MODELS = {
        "gpt-4": {"input_cost": 0.03, "output_cost": 0.06},
        "gpt-4-turbo": {"input_cost": 0.01, "output_cost": 0.03},
        "gpt-3.5-turbo": {"input_cost": 0.0015, "output_cost": 0.002},
        "gpt-3.5-turbo-16k": {"input_cost": 0.003, "output_cost": 0.004}
    }
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", default_config: Dict[str, Any] = None):
        super().__init__(api_key, model, default_config)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        if not self.validate_model(model):
            raise ValueError(f"Unsupported OpenAI model: {model}")
    
    def get_provider(self) -> ModelProvider:
        return ModelProvider.OPENAI
    
    def validate_model(self, model: str) -> bool:
        return model in self.SUPPORTED_MODELS
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using OpenAI API."""
        start_time = time.time()
        
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            response = await self.client.chat.completions.create(
                model=request.model or self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences
            )
            
            latency = time.time() - start_time
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "request_metadata": request.metadata or {}
                },
                timestamp=start_time,
                latency=latency
            )
            
            self._update_usage_stats(llm_response)
            return llm_response
            
        except Exception as e:
            latency = time.time() - start_time
            error_response = LLMResponse(
                content="",
                model=request.model or self.model,
                provider="openai",
                usage={},
                metadata={"request_metadata": request.metadata or {}},
                timestamp=start_time,
                latency=latency,
                error=str(e)
            )
            
            self._update_usage_stats(error_response)
            self.logger.error(f"OpenAI API error: {e}")
            return error_response
    
    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests concurrently."""
        tasks = [self.generate(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


class AnthropicInterface(LLMInterface):
    """Interface for Anthropic Claude models."""
    
    SUPPORTED_MODELS = {
        "claude-3-opus-20240229": {"input_cost": 0.015, "output_cost": 0.075},
        "claude-3-sonnet-20240229": {"input_cost": 0.003, "output_cost": 0.015},
        "claude-3-haiku-20240307": {"input_cost": 0.00025, "output_cost": 0.00125},
        "claude-2.1": {"input_cost": 0.008, "output_cost": 0.024},
        "claude-2.0": {"input_cost": 0.008, "output_cost": 0.024}
    }
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", default_config: Dict[str, Any] = None):
        super().__init__(api_key, model, default_config)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        if not self.validate_model(model):
            raise ValueError(f"Unsupported Anthropic model: {model}")
    
    def get_provider(self) -> ModelProvider:
        return ModelProvider.ANTHROPIC
    
    def validate_model(self, model: str) -> bool:
        return model in self.SUPPORTED_MODELS
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using Anthropic API."""
        start_time = time.time()
        
        try:
            # Anthropic uses system parameter separately
            kwargs = {
                "model": request.model or self.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            
            if request.stop_sequences:
                kwargs["stop_sequences"] = request.stop_sequences
            
            response = await self.client.messages.create(**kwargs)
            
            latency = time.time() - start_time
            
            llm_response = LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                metadata={
                    "stop_reason": response.stop_reason,
                    "request_metadata": request.metadata or {}
                },
                timestamp=start_time,
                latency=latency
            )
            
            self._update_usage_stats(llm_response)
            return llm_response
            
        except Exception as e:
            latency = time.time() - start_time
            error_response = LLMResponse(
                content="",
                model=request.model or self.model,
                provider="anthropic",
                usage={},
                metadata={"request_metadata": request.metadata or {}},
                timestamp=start_time,
                latency=latency,
                error=str(e)
            )
            
            self._update_usage_stats(error_response)
            self.logger.error(f"Anthropic API error: {e}")
            return error_response
    
    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests concurrently."""
        tasks = [self.generate(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


class LocalLLMInterface(LLMInterface):
    """Interface for local LLM endpoints (e.g., Ollama, vLLM, etc.)."""
    
    def __init__(self, api_key: str, model: str, endpoint_url: str, default_config: Dict[str, Any] = None):
        super().__init__(api_key, model, default_config)
        self.endpoint_url = endpoint_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def get_provider(self) -> ModelProvider:
        return ModelProvider.LOCAL
    
    def validate_model(self, model: str) -> bool:
        # For local models, we assume any model name is valid
        return True
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using local LLM endpoint."""
        start_time = time.time()
        
        try:
            payload = {
                "model": request.model or self.model,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            if request.system_prompt:
                payload["system"] = request.system_prompt
            
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences
            
            response = await self.client.post(
                f"{self.endpoint_url}/v1/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            llm_response = LLMResponse(
                content=data["choices"][0]["text"],
                model=data.get("model", request.model or self.model),
                provider="local",
                usage=data.get("usage", {}),
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "request_metadata": request.metadata or {}
                },
                timestamp=start_time,
                latency=latency
            )
            
            self._update_usage_stats(llm_response)
            return llm_response
            
        except Exception as e:
            latency = time.time() - start_time
            error_response = LLMResponse(
                content="",
                model=request.model or self.model,
                provider="local",
                usage={},
                metadata={"request_metadata": request.metadata or {}},
                timestamp=start_time,
                latency=latency,
                error=str(e)
            )
            
            self._update_usage_stats(error_response)
            self.logger.error(f"Local LLM API error: {e}")
            return error_response
    
    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests concurrently."""
        tasks = [self.generate(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


class LLMManager:
    """Manager class for handling multiple LLM interfaces."""
    
    def __init__(self):
        self.interfaces: Dict[str, LLMInterface] = {}
        self.default_interface: Optional[str] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_interface(self, name: str, interface: LLMInterface, set_as_default: bool = False):
        """Add an LLM interface to the manager."""
        self.interfaces[name] = interface
        if set_as_default or self.default_interface is None:
            self.default_interface = name
        self.logger.info(f"Added LLM interface '{name}' ({interface.get_provider().value})")
    
    def get_interface(self, name: Optional[str] = None) -> LLMInterface:
        """Get an LLM interface by name or default."""
        if name is None:
            name = self.default_interface
        
        if name not in self.interfaces:
            raise ValueError(f"LLM interface '{name}' not found")
        
        return self.interfaces[name]
    
    def list_interfaces(self) -> List[str]:
        """List all available interface names."""
        return list(self.interfaces.keys())
    
    async def generate(self, request: LLMRequest, interface_name: Optional[str] = None) -> LLMResponse:
        """Generate using specified or default interface."""
        interface = self.get_interface(interface_name)
        return await interface.generate(request)
    
    async def generate_with_fallback(self, request: LLMRequest, interface_names: List[str]) -> LLMResponse:
        """Try multiple interfaces in order until one succeeds."""
        last_error = None
        
        for interface_name in interface_names:
            try:
                interface = self.get_interface(interface_name)
                response = await interface.generate(request)
                
                if response.error is None:
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Interface '{interface_name}' failed: {e}")
        
        # All interfaces failed
        return LLMResponse(
            content="",
            model=request.model,
            provider="fallback",
            usage={},
            metadata={"request_metadata": request.metadata or {}},
            timestamp=time.time(),
            latency=0.0,
            error=f"All interfaces failed. Last error: {last_error}"
        )
    
    def get_combined_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all interfaces."""
        return {name: interface.get_usage_stats() for name, interface in self.interfaces.items()}


# Factory functions for easy interface creation
def create_openai_interface(api_key: str, model: str = "gpt-3.5-turbo", **kwargs) -> OpenAIInterface:
    """Create an OpenAI interface with the specified configuration."""
    return OpenAIInterface(api_key, model, kwargs)


def create_anthropic_interface(api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs) -> AnthropicInterface:
    """Create an Anthropic interface with the specified configuration."""
    return AnthropicInterface(api_key, model, kwargs)


def create_local_interface(api_key: str, model: str, endpoint_url: str, **kwargs) -> LocalLLMInterface:
    """Create a local LLM interface with the specified configuration."""
    return LocalLLMInterface(api_key, model, endpoint_url, kwargs)


def create_llm_manager_from_config(config: Dict[str, Any]) -> LLMManager:
    """Create an LLM manager from configuration dictionary."""
    manager = LLMManager()
    
    for name, interface_config in config.get("interfaces", {}).items():
        provider = interface_config["provider"]
        api_key = interface_config["api_key"]
        model = interface_config["model"]
        
        if provider == "openai":
            interface = create_openai_interface(api_key, model, **interface_config.get("config", {}))
        elif provider == "anthropic":
            interface = create_anthropic_interface(api_key, model, **interface_config.get("config", {}))
        elif provider == "local":
            endpoint_url = interface_config["endpoint_url"]
            interface = create_local_interface(api_key, model, endpoint_url, **interface_config.get("config", {}))
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        is_default = interface_config.get("default", False)
        manager.add_interface(name, interface, is_default)
    
    return manager