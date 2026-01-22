import os
import json
import time
import yaml
import random
import requests

from typing import Optional
from glob import glob


# API setting constants
API_MAX_RETRY = 8
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    """
    Chat completion using OpenAI-compatible API.
    
    Supports both official OpenAI API and local vLLM endpoints.
    API configuration is loaded from config.yaml.
    
    Args:
        model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: Optional API configuration dict with base_url and api_key.
                  If None, loads from config.yaml static_judge settings.
        
    Returns:
        Generated text response
    """
    import openai
    from util.config_loader import config
    
    # Initialize client based on config
    if api_dict:
        api_key = api_dict.get("api_key")
        base_url = api_dict.get("base_url")
    else:
        # Load from config.yaml static_judge settings
        cfg = config.static_judge_config
        api_key = cfg.get('api_key')
        base_url = cfg.get('base_url')
    
    # assert api_key is not None and base_url is not None, "API key and base URL must be provided"
    
    if base_url:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = openai.OpenAI(api_key=api_key)
    
    output = API_ERROR_OUTPUT
    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(f"Rate limit error (attempt {attempt + 1}/{API_MAX_RETRY}): {e}")
            time.sleep(API_RETRY_SLEEP * (attempt + 1))  # Increasing backoff
        except openai.BadRequestError as e:
            print(f"Bad request error: {e}")
            print(f"Messages: {messages}")
            break
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature=0.0, max_tokens=2048, api_dict=None, index=0):
    """
    Chat completion using OpenAI-compatible API.
    
    This function name is kept for backward compatibility but now uses standard OpenAI API.
    API configuration is loaded from config.yaml static_judge settings.
    
    Args:
        model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: Optional API configuration dict. If None, uses static_judge config.
        index: Index for API selection (unused, kept for compatibility)
        
    Returns:
        Generated text response
    """
    import openai
    from util.azure_gpt4o import Openai, _get_api_infos_from_config
    
    output = API_ERROR_OUTPUT
    
    try:
        # Initialize client with static judge configuration from config.yaml
        if api_dict:
            apis = [api_dict]
        else:
            apis = _get_api_infos_from_config('static')
        oai_client = Openai(apis=apis)
        
        # Check if this is a reasoning model (o1, o1-preview, etc.)
        if any(reasoning_indicator in model.lower() for reasoning_indicator in ['o1', 'o3', 'gpt-5', 'reasoning']):
            output = oai_client.gpt5_call(
                conv=messages,
                max_tokens=max_tokens,
                reasoning_effort="low"
            )
        else:
            # Standard chat completion
            output = oai_client.get_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
    except Exception as e:
        print(f"Error in chat completion: {type(e).__name__}: {e}")
    
    return output

def chat_completion_openai_azure_image(model, messages, temperature=0.0, max_tokens=2048, api_dict=None, index=0):
    """
    Chat completion with image support using OpenAI-compatible API.
    
    This function name is kept for backward compatibility but now uses standard OpenAI API.
    API configuration is loaded from config.yaml static_judge settings.
    
    Args:
        model: Model name
        messages: List of message dictionaries (can include image_url content)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: Optional API configuration dict. If None, uses static_judge config.
        index: Index for API selection (unused, kept for compatibility)
        
    Returns:
        Generated text response
    """
    import openai
    from util.azure_gpt4o import Openai, _get_api_infos_from_config
    
    try:
        # Initialize client with static judge configuration from config.yaml
        if api_dict:
            apis = [api_dict]
        else:
            apis = _get_api_infos_from_config('static')
        oai_client = Openai(apis=apis)
        output = oai_client.get_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return output
    except Exception as e:
        print(f"Error in image chat completion: {type(e).__name__}: {e}")
        return API_ERROR_OUTPUT




def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    """
    Chat completion using Anthropic API.
    
    Args:
        model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: API configuration dict with 'api_key'. Required.
        
    Returns:
        Generated text response
    """
    import anthropic

    if api_dict:
        api_key = api_dict.get("api_key")
    else:
        api_key = None
    
    if not api_key:
        raise ValueError("Anthropic API key must be provided in api_dict")

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens, api_dict=None):
    """
    Chat completion using Mistral API.
    
    Args:
        model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: API configuration dict with 'api_key'. Required.
        
    Returns:
        Generated text response
    """
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    if api_dict:
        api_key = api_dict.get("api_key")
    else:
        api_key = None
    
    if not api_key:
        raise ValueError("Mistral API key must be provided in api_dict")
    
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens, api_dict=None):
    """
    Chat completion using Google Gemini API.
    
    Args:
        model: Model name
        message: Text message
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: API configuration dict with 'api_key'. Required.
        
    Returns:
        Generated text response
    """
    if api_dict:
        api_key = api_dict.get("api_key")
    else:
        api_key = None
    
    if not api_key:
        raise ValueError("Gemini API key must be provided in api_dict")
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output


def chat_completion_cohere(model, messages, temperature, max_tokens, api_dict=None):
    """
    Chat completion using Cohere API.
    
    Args:
        model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_dict: API configuration dict with 'api_key'. Required.
        
    Returns:
        Generated text response
    """
    import cohere

    if api_dict:
        api_key = api_dict.get("api_key")
    else:
        api_key = None
    
    if not api_key:
        raise ValueError("Cohere API key must be provided in api_dict")

    co = cohere.Client(api_key)
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
