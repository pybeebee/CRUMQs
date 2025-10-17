import os
import sys
import contextlib

import logging
import time
import numpy as np
from collections import defaultdict
from random import uniform
from typing import Any, List, Optional, Dict, Tuple
from tqdm.asyncio import tqdm

from langchain_core.language_models import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield   # Do nothing
    else:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

class AnswerMatchMode:
    """How to match the model response to YES/NO.

    Supported options:
    - STARTS_WITH: Match if the response starts with YES/NO
    - CONTAINS: Match if the response contains YES/NO
    """

    STARTS_WITH = "starts_with"
    CONTAINS = "contains"


class ModelProvider:
    """Model provider configuration.

    Supported providers:
    - OPENAI: OpenAI's models (e.g., GPT-4)
    - ANTHROPIC: Anthropic's models (e.g., Claude)
    - TOGETHER: Together.ai's models (e.g., Llama)
    - HUGGINGFACE: Hugging Face's models (e.g., Llama)
    - NULL: Special value reserved for purely automatic checks
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_LOCAL = "huggingface_local"
    LITELLM_LOCAL = "litellm_local"
    NULL = "null"


DEFAULT_MODELS_BY_PROVIDER = {
    ModelProvider.OPENAI: "gpt-4o-mini-2024-07-18",
    ModelProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    ModelProvider.TOGETHER: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ModelProvider.HUGGINGFACE: "meta-llama/Llama-3.3-70B-Instruct",
    ModelProvider.HUGGINGFACE_LOCAL: "meta-llama/Llama-3.1-8B-Instruct",
    ModelProvider.LITELLM_LOCAL: "meta-llama/Llama-3.3-70B-Instruct",
}

# Cache for model clients to avoid needing to reinstantiate on each request
CLIENT_CACHE = defaultdict(dict)


def load_model(
        provider: str=ModelProvider.TOGETHER, 
        model_name: str=None,
        max_tokens: int=500,
        temperature: float=None,
        top_p: float=None,
        stop: List[str]=None,
    ): # -> BaseChatModel
    """
    Get the appropriate model based on provider.

    More parameters: https://python.langchain.com/docs/concepts/chat_models/#standard-parameters

    Args:
        provider: The model provider to use (openai, anthropic, or together)
        model_name: Optional specific model name to use
        max_tokens: Maximum output tokens
        temperature: Temperature for temperature sampling
        top_p: Probability threshold for nucleus sampling
        stop: Stop sequences (strings)

    Returns:
        A configured LangChain chat model instance

    Raises:
        ValueError: If an unsupported provider is specified
    """
    
    # Get default model if not provided
    if not model_name:
        model_name = DEFAULT_MODELS_BY_PROVIDER[provider]

    # Use cached model if there (local HF only)
    if model_name in CLIENT_CACHE[provider]:
        return CLIENT_CACHE[provider][model_name]

    # Load OpenAI model
    if provider == ModelProvider.OPENAI:
        client = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature, 
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
    
    # Load Anthropic model
    elif provider == ModelProvider.ANTHROPIC:
        client = ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
    
    # Load TogetherAI model
    elif provider == ModelProvider.TOGETHER:
        client = ChatTogether(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
    
    # Load HF model (HF API)
    elif provider == ModelProvider.HUGGINGFACE:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        client = ChatHuggingFace(llm=llm)
    
    # Load HF model (local)
    elif provider == ModelProvider.HUGGINGFACE_LOCAL:
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "temperature": temperature,
                "top_p": top_p,
            },
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        client = ChatHuggingFace(llm=llm, tokenizer=tokenizer, stop=stop)
    
    # Load local LLM (LiteLLM)
    elif provider == ModelProvider.LITELLM_LOCAL:
        client = ChatOpenAI(
            model_name=model_name, # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_IP"),
        )
    
    # Error if provider invalid
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    
    CLIENT_CACHE[provider][model_name] = client

    return client


async def get_response(
        system_prompt: str,
        user_prompt: str,
        
        provider: str=ModelProvider.TOGETHER,
        model_name: str=None,
        max_tokens: int=500,
        temperature: float=None,
        top_p: float=None,
        stop: List[str]=None,
        max_retries: int=1,
        strip: str=None,
        
        base_delay: float=2.0,
        use_cache: bool=True,
    ) -> str:
    """
    Get a response from the specified model with retry logic.

    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        provider: The model provider to use
        model_name: Optional specific model name
        max_tokens: Maximum output tokens
        temperature: Temperature for temperature sampling
        top_p: Probability threshold for nucleus sampling
        stop: Stop sequences (strings)
        max_retries: Maximum number of retries on failure
        base_delay: Base delay between retries (uses exponential backoff)
        use_cache: If allowing LLM API server to use cache

    Returns:
        The model's response

    Raises:
        RuntimeError: If max retries exceeded or invalid response received
    """
    for attempt in range(max_retries):
        try:
            # Load model
            model = load_model(
                provider=provider,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            
            # Disable cache if specified
            extra_body = {}
            if not use_cache:
                extra_body['cache'] = {'no-cache': True, 'no-store': True}

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await model.ainvoke(messages, extra_body=extra_body)
            response_text = response.content.strip()
            if strip:
                response_text = response_text.strip(strip)
            return response_text
        
        except ValueError:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        except Exception as e:
            
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            elif attempt < max_retries - 1:
                delay = base_delay + uniform(0, 0.1)
                logger.warning(f"Error: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            raise RuntimeError(
                f"Model response error after {max_retries} attempts: {str(e)}"
            )


async def get_response_batch(
        system_prompt: str,
        user_prompts: List[str],
        
        provider: str = ModelProvider.TOGETHER,
        model_name: str = None,
        max_tokens: int=500,
        temperature: float=None,
        top_p: float=None,
        stop: List[str]=None,
        max_retries: int = 3,

        strip: str=None,
        uppercase: bool=False,
        modify_response: bool=False,
        
        base_delay: float = 2.0,
        response_on_error: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[str]:
    """
    Get multiple responses from the model in batches.

    Processes prompts in batches to avoid rate limits while maintaining efficiency.

    Args:
        system_prompt: The system prompt to use for all queries
        user_prompts: List of user prompts to process
        provider: The model provider to use
        model_name: Optional specific model name
        max_tokens: Maximum output tokens
        temperature: Temperature for temperature sampling
        top_p: Probability threshold for nucleus sampling
        stop: Stop sequences (strings)
        max_retries: Maximum number of retries on failure
        uppercase: Whether to uppercase-ify the responses
        base_delay: Base delay between retries (uses exponential backoff)
        response_on_error: Optional response to return on malformed responses
        use_cache: If allowing LLM API server to use cache

    Returns:
        List of responses matching the input prompts order

    Note:
        Uses batching to reduce API calls and includes rate limit handling.
        A small delay is added between batches to avoid overwhelming the API.
    """
    
    for attempt in range(max_retries):
        try:
            
            # Load model
            model = load_model(
                provider=provider,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            
            # Disable cache if specified
            extra_body = {}
            if not use_cache:
                extra_body['cache'] = {'no-cache': True, 'no-store': True}
            raw_responses = await tqdm.gather(
                *[
                    model.ainvoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt),
                        ],
                        extra_body=extra_body,
                    )
                    for user_prompt in user_prompts
                ],
                desc="Processing prompts",
            )
            response_text = [response.content.strip() for response in raw_responses]
            
            if strip: 
                response_text = [response.strip(strip) for response in response_text]

            if uppercase:
                response_text = [response.upper() for response in response_text]

            if modify_response:
                response_text = [
                    modify_model_response(
                        response,
                        response_on_error=response_on_error
                    )[0]
                    for response in response_text
                ]

            return response_text

        except Exception as e:
            
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            elif attempt < max_retries - 1:
                delay = base_delay + uniform(0, 0.1)
                logger.warning(f"Error: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            raise RuntimeError(
                f"Model response error after {max_retries} attempts: {str(e)}"
            )


def modify_model_response(
        response_text: str,
        answer_match_mode: str = AnswerMatchMode.CONTAINS,
        response_on_error: Optional[str] = None,
    ): # -> Tuple[str, bool]
    """
    Modify model response to be VITAL/OKAY.

    Args:
        response_text: The model output to verify
        answer_match_mode: How to match the response to VITAL/OKAY
        response_on_error: Optional response to return on malformed
            (non-VITAL/OKAY) responses (defaults to None)

    Returns:
        The model's response as "VITAL" or "OKAY"

    Raises:
        ValueError: If model response is malformed
    """

    if answer_match_mode == AnswerMatchMode.STARTS_WITH:
        if len(response_text) > 3:
            if response_text.startswith("VITAL"):
                response_text = "VITAL"
            elif response_text.startswith("OKAY"):
                response_text = "OKAY"
   
    elif answer_match_mode == AnswerMatchMode.CONTAINS:
        if "VITAL" in response_text:
            response_text = "VITAL"
        elif "OKAY" in response_text:
            response_text = "OKAY"
   
    else:
        raise ValueError(f"Unsupported answer match mode: {answer_match_mode}")

    has_error = False
    if response_text not in ["VITAL", "OKAY"]:
        if response_on_error is not None:
            logger.warning(
                f"Invalid model response: {response_text}. Returning default response: {response_on_error}"
            )
            response_text = response_on_error
            has_error = True
        else:
            raise ValueError(
                f"Invalid model response: {response_text}. Expected VITAL or OKAY."
            )

    return response_text, has_error

