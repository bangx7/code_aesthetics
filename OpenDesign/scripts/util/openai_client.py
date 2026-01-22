"""
Standard OpenAI API client for OpenDesign Benchmark.

This module provides a unified interface for interacting with OpenAI-compatible APIs,
supporting both official OpenAI endpoints and other compatible services.
"""

import os
import time
import base64
from typing import List, Dict, Optional, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    A wrapper for OpenAI API that supports standard OpenAI and compatible endpoints.
    
    This client automatically retries on rate limit errors and provides image handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 5,
        timeout: int = 60
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (must be provided, no env var fallback)
            base_url: Base URL for API (None for official OpenAI endpoint)
            model: Model name to use
            max_retries: Maximum number of retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("API key must be provided in config.yaml")
        
        # Initialize OpenAI client
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Create a chat completion with automatic retry on rate limits.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # Retry on rate limit or temporary errors
                if "rate" in str(e).lower() or "RateLimitError" in error_type:
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    retry_count += 1
                elif "timeout" in str(e).lower() or "TimeoutError" in error_type:
                    wait_time = 5
                    logger.warning(f"Timeout error, retrying {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    # Non-retryable error
                    logger.error(f"API error: {error_type}: {str(e)}")
                    raise
        
        # Max retries exceeded
        logger.error(f"Max retries ({self.max_retries}) exceeded. Last error: {last_error}")
        raise last_error
    
    def chat_completion_with_image(
        self,
        text: str,
        image_path: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Create a chat completion with an image input.
        
        Args:
            text: Text prompt
            image_path: Path to image file
            system_prompt: System message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        base64_image = self.encode_image(image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        return self.chat_completion(messages, temperature, max_tokens, **kwargs)


def create_client_from_config(config: Dict[str, Any]) -> OpenAIClient:
    """
    Create OpenAI client from configuration dictionary.
    
    Args:
        config: Configuration dictionary with api_key, base_url, model, etc.
        
    Returns:
        Configured OpenAIClient instance
    """
    return OpenAIClient(
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        model=config.get("model", "gpt-4o"),
        max_retries=config.get("max_retries", 5),
        timeout=config.get("timeout", 60)
    )


if __name__ == "__main__":
    # Example usage
    client = OpenAIClient(model="gpt-4o")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, World!'"}
    ]
    response = client.chat_completion(messages)
    print(response)

