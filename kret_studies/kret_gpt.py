import socket
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM

# Use the actual class used by transformers for type checking
from transformers.models.auto.modeling_auto import AutoModelForCausalLM as HF_AutoModelForCausalLM


# === CONFIGURATION ===
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable Hugging Face offline mode if requested
TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0")
if TRANSFORMERS_OFFLINE == "1":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("[Transformers] Offline mode enabled: will not attempt to download models or tokenizers.")

# GPU-aware model selection
HAS_CUDA = torch.cuda.is_available()
LOCAL_MODEL_NAME = "justinthelaw/Hermes-2-Pro-Mistral-7B-4bit-32g-GPTQ" if HAS_CUDA else "tiiuae/falcon-rw-1b"
REMOTE_MODEL_NAME = "microsoft/mai-ds-r1:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in your environment

GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Default Gemini model for text generation
GEMINI_KEY = os.getenv("GEMINI_KEY")  # Set this in your environment

# === LOCAL MODEL LOADING (LAZY, TYPE-SAFE) ===

_local_tokenizer: PreTrainedTokenizerBase | None = None
_kret_model: HF_AutoModelForCausalLM | None = None


def load_local_model() -> tuple[PreTrainedTokenizerBase, HF_AutoModelForCausalLM]:
    global _local_tokenizer, _kret_model
    if _local_tokenizer is None or _kret_model is None:
        print("🔄 Loading local model...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, use_fast=True)
        _kret_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, device_map="auto", trust_remote_code=True)
        # Suppress static analysis warning for .eval()
        if _kret_model is not None and hasattr(_kret_model, "eval"):
            getattr(_kret_model, "eval")()
    # Ensure loaded
    assert _local_tokenizer is not None and _kret_model is not None, "Local model or tokenizer not loaded!"
    tokenizer = _local_tokenizer
    model = _kret_model
    return tokenizer, model


# === LOCAL QUERY ===
def query_local(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = load_local_model()
    # No type assertions: allow any compatible model
    inputs = tokenizer(prompt, return_tensors="pt")
    # Suppress static analysis warning for .device
    device = getattr(model, "device", None)
    if device is not None:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
    with torch.no_grad():
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs["attention_mask"]
        # Suppress static analysis warning for .generate
        outputs = getattr(model, "generate")(**generate_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# === REMOTE QUERY ===
def query_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": REMOTE_MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# === REMOTE QUERY (Google AI - Gemini) ===
def query_gemini(prompt: str) -> str:
    """
    Queries the Google AI (Gemini) API.
    """
    global GEMINI_KEY  # Ensure we can access the global GEMINI_KEY
    GEMINI_KEY = os.getenv("GEMINI_KEY")  # Re-fetch just in case it was set after script start

    if not GEMINI_KEY:
        raise ValueError("Missing GEMINI_KEY. Set GEMINI_KEY in your environment.")

    # Construct the API URL using the model name and API key
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_KEY}"
    )

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    # Parse the JSON response and extract the content
    result = response.json()
    if result and "candidates" in result and result["candidates"]:
        first_candidate = result["candidates"][0]
        if "content" in first_candidate and "parts" in first_candidate["content"]:
            first_part = first_candidate["content"]["parts"][0]
            if "text" in first_part:
                return first_part["text"]

    raise ValueError(f"Unexpected response format from Gemini API: {result}")


# === UNIVERSAL INTERFACE ===


def has_internet(host="8.8.8.8", port=53, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def query_llm(prompt: str, use_local_override: bool | None = None, use_gemini=True) -> str:
    use_local = use_local_override if use_local_override is not None else not has_internet()
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if use_local:
        print("Using local model...")
        return query_local(prompt)
    else:
        if use_gemini:
            print("Using Gemini API...")
            return query_gemini(prompt)
        else:
            print("Using OpenRouter API...")
            return query_openrouter(prompt)
