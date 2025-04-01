import os
import logging
from typing import Dict, Any, Optional, Union, Callable

# Import different LLM providers
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory class for creating and configuring LLM instances based on provider specifications."""
    
    # Default API keys - will be populated on first use
    _default_api_keys = {}
    
    @classmethod
    def _get_api_key(cls, provider: str, config_api_key: Optional[str] = None) -> str:
        """
        Get the API key for a provider using the following priority:
        1. Explicitly provided API key
        2. API key from app_config.json
        3. API key from environment variables
        4. Default API key (if previously loaded)
        
        Args:
            provider: The LLM provider name (openai, google, deepseek)
            config_api_key: API key provided in the app_config.json
            
        Returns:
            API key string
            
        Raises:
            ValueError: If no API key can be found for the provider
        """
        provider = provider.lower()
        
        # 1. If explicitly provided, use it
        if config_api_key:
            # Cache this key as a default for future use
            cls._default_api_keys[provider] = config_api_key
            return config_api_key
            
        # 2. Check cached default keys
        if provider in cls._default_api_keys and cls._default_api_keys[provider]:
            return cls._default_api_keys[provider]
        
        # 3. Try to get API key from environment variables
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var_name)
        
        if api_key:
            # Cache this key as a default for future use
            cls._default_api_keys[provider] = api_key
            return api_key
            
        # 4. Special case for DeepSeek to try DeepInfra token
        if provider == "deepseek":
            deepinfra_key = os.getenv("DEEPINFRA_API_TOKEN")
            if deepinfra_key:
                cls._default_api_keys[provider] = deepinfra_key
                return deepinfra_key
                
        # No key found
        raise ValueError(f"API key not provided for {provider}. Please set {env_var_name} environment variable or provide it in app_config.json")
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create and configure an LLM instance based on the specified provider.
        
        Args:
            provider: The LLM provider name (openai, google, deepseek)
            model_name: The specific model name to use
            temperature: Temperature setting for model response randomness
            max_tokens: Maximum tokens for model response
            api_key: API key for the model provider
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A configured LLM instance
            
        Raises:
            ValueError: If provider is not supported or missing required parameters
        """
        provider = provider.lower()
        
        # Get the API key based on priority
        api_key = LLMFactory._get_api_key(provider, api_key)
        
        # Create and configure the LLM based on provider
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                **kwargs
            )
        
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                google_api_key=api_key,
                convert_system_message_to_human=True,  # Gemini-specific parameter
                **kwargs
            )
        
        elif provider == "deepseek":
            return ChatDeepSeek(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                max_retries=kwargs.get("max_retries", 2),
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are: openai, google, deepseek")

    @staticmethod
    def create_chain_for_ocr(
        provider: str = "openai",
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Create a chain specifically for OCR processing.
        
        Returns:
            A callable function that processes OCR requests
        """
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        # Create a simple chain that directly returns the content
        def run_chain(messages_dict):
            result = llm.invoke(messages_dict["messages"])
            return result.content
        
        return run_chain
    
    @staticmethod
    def create_chain_for_interpreter(
        provider: str = "google",
        model_name: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Create a chain specifically for interpreter processing.
        
        Returns:
            A callable function that processes interpreter requests
        """
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        # Return a reference to the LLM directly as the interpreter will handle the chain logic
        return llm
    
    @staticmethod
    def create_chain_for_evaluation(
        provider: str = "deepseek",
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Create a chain specifically for evaluation processing.
        
        Returns:
            A callable function that processes evaluation requests
        """
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        # Create a function that processes evaluation requests
        def run_chain(inputs):
            try:
                # Create messages in the format expected by the LLM
                if provider == "deepseek":
                    # DeepSeek expects tuples of (role, content)
                    messages = [
                        ("system", inputs["system"]),
                        ("human", inputs["user"])
                    ]
                else:
                    # Standard format for OpenAI and others
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content=inputs["system"]),
                        HumanMessage(content=inputs["user"])
                    ]
                
                # Invoke the model with proper message format
                result = llm.invoke(messages)
                
                # Extract and return the content
                if hasattr(result, 'content'):
                    return result.content
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error in evaluation chain: {str(e)}")
                return f"Error in evaluation: {str(e)}"
        
        return run_chain 