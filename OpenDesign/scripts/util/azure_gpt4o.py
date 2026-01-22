"""
OpenAI-compatible API client for OpenDesign Benchmark.

This module provides a simple wrapper around the OpenAI API client
to support standard OpenAI endpoints and other compatible services.
"""

import random
import time
import base64
import os
from openai import OpenAI
import openai

# Import config loader to get API settings from config.yaml
from util.config_loader import config


def _get_api_infos_from_config(judge_type='interactive'):
    """
    Get API configuration from config.yaml based on judge type.
    
    Args:
        judge_type: 'static' for static_judge, 'interactive' for interactive_judge,
                   'openai' for answer generation
    
    Returns:
        List of API configuration dictionaries
    """
    if judge_type == 'static':
        cfg = config.static_judge_config
    elif judge_type == 'interactive':
        cfg = config.interactive_judge_config
    else:  # 'openai' or default
        cfg = config.openai_config
    
    return [{
        "model": cfg.get('model', 'gpt-4o'),
        "api_key": cfg.get('api_key'),
        "base_url": cfg.get('base_url'),
    }]


# Default API_INFOS uses interactive_judge config (for backward compatibility with agent_score)
API_INFOS = _get_api_infos_from_config('interactive')


def get_API_list(judge_type='interactive'):
    """
    Get list of API configurations.
    
    Args:
        judge_type: 'static' for static_judge, 'interactive' for interactive_judge
    """
    return _get_api_infos_from_config(judge_type)


class Openai:
    """
    OpenAI-compatible API client wrapper.
    
    This class provides a simple interface for making API calls to OpenAI
    or compatible services like vLLM, with automatic retry on rate limits.
    """
    
    def __init__(self, apis=None):
        """
        Initialize OpenAI client.
        
        Args:
            apis: List of API configuration dicts. If None, uses default API_INFOS.
                  Each dict should contain: model, api_key, base_url (optional)
        """
        if apis is None:
            apis = API_INFOS
        
        # Select first available API configuration
        selected_api = apis[0] if apis else API_INFOS[0]
        
        self.model = selected_api.get('model', 'gpt-4o')
        api_key = selected_api.get('api_key')
        if not api_key:
            raise ValueError("API key must be provided in config.yaml (openai.api_key, static_judge.api_key, or interactive_judge.api_key)")
        base_url = selected_api.get('base_url')
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=10,
                timeout=60.0
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                max_retries=10,
                timeout=60.0
            )
 
    def encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_response_v2_raw(self, content, image, system="You are a helpful assistant.", max_tokens=2048, client_index=None):
        """
        Get response with image input.
        
        Args:
            content: Text prompt
            image: Path to image file
            system: System message
            max_tokens: Maximum tokens to generate
            client_index: Unused (kept for compatibility)
            
        Returns:
            Tuple of (response_text, finish_reason)
        """
        base64_image = self.encode_image(image)
        
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            },
        ]
        
        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                results = completion.choices[0].message.content
                stop_reason = completion.choices[0].finish_reason
                return results, stop_reason
            except openai.RateLimitError:
                time.sleep(2 ** cur_retry)  # Exponential backoff
                cur_retry += 1
            except Exception as e:
                print(f"Error in image response: {e}")
                cur_retry += 1
        
        return "", ""      
 
    def get_image_response_v3_raw(self, content, system="You are a helpful assistant.", max_tokens=2048, client_index=None):
        """
        Get response with pre-formatted messages (for multi-image inputs).
        
        Args:
            content: Pre-formatted messages list
            system: System message (unused if content already has system message)
            max_tokens: Maximum tokens to generate
            client_index: Unused (kept for compatibility)
            
        Returns:
            Tuple of (response_text, finish_reason)
        """
        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=content,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                results = completion.choices[0].message.content
                stop_reason = completion.choices[0].finish_reason
                return results, stop_reason
            except openai.RateLimitError:
                time.sleep(2 ** cur_retry)  # Exponential backoff
                cur_retry += 1
            except Exception as e:
                print(f"Error in image response v3: {e}")
                cur_retry += 1
        
        return "", ""  
   
    def get_response(self, messages, max_tokens=2048, temperature=0.0):
        """
        Get chat completion response.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response text string
        """
        max_retry = 15
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            except openai.RateLimitError:
                wait_time = min(2 ** cur_retry, 30)  # Exponential backoff, max 30s
                time.sleep(wait_time)
                cur_retry += 1
            except Exception as e:
                print(f"Error in get_response: {e}")
                cur_retry += 1
        
        return ""
           
    def call(self, content, max_tokens=16380, temperature=0.7, frequency_penalty=0, presence_penalty=0, seed_param=None):
        """
        Simple text completion call.
        
        Args:
            content: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            seed_param: Random seed for reproducibility
            
        Returns:
            Response text string
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        
        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Add optional parameters if provided
                if frequency_penalty:
                    kwargs["frequency_penalty"] = frequency_penalty
                if presence_penalty:
                    kwargs["presence_penalty"] = presence_penalty
                if seed_param is not None:
                    kwargs["seed"] = seed_param
                
                completion = self.client.chat.completions.create(**kwargs)
                return completion.choices[0].message.content
            except openai.RateLimitError:
                time.sleep(2 ** cur_retry)  # Exponential backoff
                cur_retry += 1
            except Exception as e:
                print(f"Error in call: {e}")
                cur_retry += 1
        
        return ""
    
    def gpt5_call(self, conv, max_tokens, reasoning_effort="low"):
        """
        Call reasoning models (e.g., o1, o1-preview) with reasoning effort parameter.
        
        Args:
            conv: Pre-formatted conversation messages
            max_tokens: Maximum completion tokens
            reasoning_effort: Reasoning effort level (low/medium/high)
            
        Returns:
            Response text string
        """
        max_retry = 10
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                # For reasoning models, use max_completion_tokens instead of max_tokens
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=conv,
                    max_completion_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
                return completion.choices[0].message.content
            except openai.RateLimitError:
                wait_time = min(2 ** cur_retry, 60)  # Exponential backoff, max 60s
                time.sleep(wait_time)
                cur_retry += 1
            except Exception as e:
                print(f"Error in gpt5_call: {e}")
                cur_retry += 1
        
        return ""


if __name__ == "__main__":
    # Example usage
    client = Openai()
    
    # Test simple text completion
    response = client.call("Hello, who are you?")
    print(f"Text response: {response}")
    
    # Test with messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    response = client.get_response(messages)
    print(f"Chat response: {response}")