
"""
Configuration file for LLM model parameters
"""

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    # Model selection
    "model_name": "openrouter/google/gemini-2.0-flash-001",
    
    # Generation parameters
    "temperature": 0.2,          # Lower values (0.0-0.3) for factual/consistent responses
                                 # Higher values (0.7-1.0) for creative responses
    
    "top_p": 0.3,               # Controls token selection diversity 
                                # Lower values focus on high-probability tokens
    
    # Token limits
    "max_tokens": 4096,         # Maximum output tokens to generate
    "max_input_tokens": 32000,  # Not directly used in API calls but for context building
    
    # Search configuration
    "max_sources": 2,           # Number of sources to process from search results
    "reranker": "jina",         # Reranker to use (options: "jina", "infinity")
    "search_provider": "serper", # Search provider (options: "serper", "searxng")
    
    # Advanced options
    "pro_mode": True            # Whether to use pro mode for deeper search
}

# Model-specific configurations
MODEL_CONFIGS = {
    "gemini-flash": {
        "model_name": "openrouter/google/gemini-2.0-flash-001",
        "temperature": 0.2,
        "top_p": 0.3,
        "max_tokens": 4096
    },
    "claude-sonnet": {
        "model_name": "anthropic/claude-3-7-sonnet-20250219",
        "temperature": 0.2,
        "top_p": 0.3,
        "max_tokens": 4096
    },
    "gpt4-turbo": {
        "model_name": "openai/gpt-4-turbo",
        "temperature": 0.2,
        "top_p": 0.3,
        "max_tokens": 4096
    },
    "deepseek": {
        "model_name": "fireworks_ai/accounts/fireworks/models/deepseek-r1-basic",
        "temperature": 0.2,
        "top_p": 0.3,
        "max_tokens": 4096
    }
}

def get_model_config(model_key=None, **overrides):
    """
    Get the configuration for a specific model or the default configuration.
    
    Args:
        model_key (str, optional): Key for a predefined model configuration.
        **overrides: Any parameters to override in the configuration.
        
    Returns:
        dict: The model configuration.
    """
    if model_key and model_key in MODEL_CONFIGS:
        # Start with the default config for all parameters
        config = DEFAULT_MODEL_CONFIG.copy()
        # Update with model-specific parameters
        config.update(MODEL_CONFIGS[model_key])
    else:
        config = DEFAULT_MODEL_CONFIG.copy()
    
    # Apply any overrides passed as kwargs
    if overrides:
        config.update(overrides)
        
    return config
