import os
import logging
from config import config

logger = logging.getLogger(__name__)

def setup_llama_llm():
    """Set up the selected LLM (HuggingFace, OpenAI, Gemini) using config."""
    provider = config.llm_provider.lower()
    logger.info(f"Setting up LLM provider: {provider}")

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        if not config.huggingface_api_token:
            logger.error("HUGGINGFACE_API_TOKEN not found in environment variables")
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")
        return HuggingFaceEndpoint(
            repo_id=config.llm_repo_id,
            huggingfacehub_api_token=config.huggingface_api_token,
            task="text-generation",
            temperature=config.llm_temperature,
            model_kwargs={"max_length": config.llm_max_length}
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not config.openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model_name=config.openai_model or "gpt-3.5-turbo",
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_length
        )

    elif provider == "gemini":
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not config.google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=config.google_api_key)
        return ChatGoogleGenerativeAI(
            model=config.gemini_model or "gemini-pro",
            temperature=config.llm_temperature,
            max_output_tokens=config.llm_max_length
        )

    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}") 