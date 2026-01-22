"""
Unified configuration loader for OpenDesign Benchmark.
Loads all settings from config.yaml.
"""

import yaml
import os
from pathlib import Path


class Config:
    """Singleton configuration loader."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from config.yaml."""
        # Find config.yaml in project root
        current_dir = Path(__file__).resolve().parent
        config_path = current_dir.parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Replace environment variables
        self._replace_env_vars(self._config)
    
    def _replace_env_vars(self, obj):
        """Recursively replace ${VAR} with environment variables."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                    env_var = item[2:-1]
                    obj[i] = os.getenv(env_var)
                elif isinstance(item, (dict, list)):
                    self._replace_env_vars(item)
    
    def get(self, key_path, default=None):
        """
        Get configuration value by dot-separated path.
        
        Example:
            config.get('openai.api_key')
            config.get('generation.temperature')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_all(self):
        """Get entire configuration dictionary."""
        return self._config
    
    @property
    def model_to_evaluate(self):
        return self.get('model_to_evaluate')
    
    @property
    def use_api(self):
        return self.get('use_api', True)
    
    @property
    def judge_model(self):
        """Deprecated: use static_judge_config or interactive_judge_config instead."""
        return self.get('static_judge.model', 'gpt-4o')
    
    @property
    def num_threads(self):
        return self.get('generation.num_threads', 32)
    
    @property
    def output_dir(self):
        return self.get('benchmark.output_dir', 'arena-bench-result')
    
    @property
    def openai_config(self):
        """Get OpenAI API configuration for answer generation."""
        return {
            'api_key': self.get('openai.api_key'),
            'base_url': self.get('openai.base_url'),
            'model': self.get('openai.model', 'gpt-4o'),
            'max_tokens': self.get('openai.max_tokens', 8192),
            'temperature': self.get('openai.temperature', 0.7),
        }
    
    @property
    def static_judge_config(self):
        """Get static aesthetics judge model configuration."""
        return {
            'api_key': self.get('static_judge.api_key'),
            'base_url': self.get('static_judge.base_url'),
            'model': self.get('static_judge.model', 'gpt-4o'),
            'max_tokens': self.get('static_judge.max_tokens', 4096),
            'temperature': self.get('static_judge.temperature', 0.0),
        }
    
    @property
    def interactive_judge_config(self):
        """Get interactive score judge model configuration."""
        return {
            'api_key': self.get('interactive_judge.api_key'),
            'base_url': self.get('interactive_judge.base_url'),
            'model': self.get('interactive_judge.model', 'gpt-4o'),
            'max_tokens': self.get('interactive_judge.max_tokens', 4096),
            'temperature': self.get('interactive_judge.temperature', 0.0),
        }


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test config loading
    print("Model to evaluate:", config.model_to_evaluate)
    print("Use API:", config.use_api)
    print("Judge model:", config.judge_model)
    print("Num threads:", config.num_threads)
    print("OpenAI API key set:", bool(config.get('openai.api_key')))

