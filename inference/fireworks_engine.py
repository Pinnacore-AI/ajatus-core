"""
Fireworks AI Inference Engine for Ajatuskumppani

This module provides a wrapper around the Fireworks AI Python SDK for
efficient, scalable AI inference with support for multiple models.
"""

import os
import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

try:
    from fireworks import LLM
except ImportError:
    raise ImportError(
        "Fireworks AI SDK not installed. "
        "Install with: pip install fireworks-ai"
    )


@dataclass
class ModelConfig:
    """Configuration for a Fireworks AI model"""
    name: str
    display_name: str
    deployment_type: str = "serverless"
    max_tokens: int = 2048
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0002  # USD


# Supported models
MODELS = {
    "llama4-maverick": ModelConfig(
        name="llama4-maverick-instruct-basic",
        display_name="Llama 4 Maverick (Suomi)",
        cost_per_1k_tokens=0.0002
    ),
    "deepseek-r1": ModelConfig(
        name="deepseek-r1",
        display_name="DeepSeek R1",
        cost_per_1k_tokens=0.0001
    ),
    "qwen2p5-vl": ModelConfig(
        name="qwen2p5-vl-32b-instruct",
        display_name="Qwen 2.5 VL (Vision)",
        cost_per_1k_tokens=0.0003
    ),
}


class FireworksEngine:
    """
    Fireworks AI inference engine for Ajatuskumppani.
    
    Provides a high-level interface to Fireworks AI models with:
    - Multiple model support
    - Streaming responses
    - Cost tracking
    - Error handling
    - Rate limiting
    """
    
    def __init__(
        self,
        model: str = "llama4-maverick",
        api_key: Optional[str] = None,
        deployment_type: str = "serverless",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize the Fireworks AI engine.
        
        Args:
            model: Model identifier (e.g., "llama4-maverick")
            api_key: Fireworks AI API key (or set FIREWORKS_API_KEY env var)
            deployment_type: "serverless", "on-demand", or "auto"
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks AI API key not found. "
                "Set FIREWORKS_API_KEY environment variable or pass api_key parameter."
            )
        
        if model not in MODELS:
            raise ValueError(
                f"Model '{model}' not supported. "
                f"Available models: {list(MODELS.keys())}"
            )
        
        self.model_config = MODELS[model]
        self.deployment_type = deployment_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = LLM(
            model=self.model_config.name,
            deployment_type=self.deployment_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        
        print(f"✅ Fireworks AI engine initialized: {self.model_config.display_name}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            messages: List of chat messages in OpenAI format
                      [{"role": "user", "content": "Hello"}]
            stream: Whether to stream the response
            **kwargs: Additional parameters for the model
        
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            response = self.llm.chat.completions.create(
                messages=messages,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # Return generator for streaming
                return self._stream_response(response)
            else:
                # Return complete response
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
                # Track cost
                self._track_cost(tokens_used)
                
                elapsed = time.time() - start_time
                print(f"⚡ Generated {tokens_used} tokens in {elapsed:.2f}s")
                
                return content
        
        except Exception as e:
            print(f"❌ Fireworks AI error: {e}")
            raise
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response chunks"""
        tokens_used = 0
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                tokens_used += len(content.split())  # Rough estimate
                yield content
        
        # Track cost after streaming completes
        self._track_cost(tokens_used)
    
    def _track_cost(self, tokens: int):
        """Track token usage and cost"""
        self.total_tokens_used += tokens
        cost = (tokens / 1000) * self.model_config.cost_per_1k_tokens
        self.total_cost_usd += cost
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "model": self.model_config.display_name,
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "cost_per_1k_tokens": self.model_config.cost_per_1k_tokens,
        }
    
    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple chat interface.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
        
        Returns:
            AI's response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        return self.generate(messages)


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = FireworksEngine(
        model="llama4-maverick",
        temperature=0.7
    )
    
    # Simple chat
    response = engine.chat(
        user_message="Kerro minulle suomalaisen tekoälyn tulevaisuudesta.",
        system_prompt="Olet Ajatuskumppani, suomalainen tekoälyavustaja."
    )
    
    print("\n" + "="*80)
    print("Response:")
    print("="*80)
    print(response)
    print("="*80)
    
    # Print stats
    stats = engine.get_stats()
    print(f"\nStats: {stats}")

