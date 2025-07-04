import socket
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, tokenization_utils_base

# === CONFIGURATION ===
USE_LOCAL = True  # Set to False to use remote model
LOCAL_MODEL_NAME = "TheBloke/Hermes-2-Pro-Mistral-7B-GPTQ"
REMOTE_MODEL_NAME = "deepseek-chat"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in your environment


# === LOCAL MODEL LOADING ===

_local_tokenizer: PreTrainedTokenizerBase | None = None
_kret_model: PreTrainedModel | None = None


def load_local_model():
    global _local_tokenizer, _kret_model
    if _kret_model is None or _local_tokenizer is None:
        print("🔄 Loading local model...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, use_fast=True)
        _kret_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, device_map="auto", trust_remote_code=True)
        if _kret_model is not None and hasattr(_kret_model, "eval"):
            _kret_model.eval()


# === LOCAL QUERY ===
def query_local(prompt: str, max_new_tokens: int = 256) -> str:
    load_local_model()
    assert _local_tokenizer is not None and _kret_model is not None, "Local model or tokenizer not loaded!"
    inputs = _local_tokenizer(prompt, return_tensors="pt")
    if hasattr(_kret_model, "device"):
        for k, v in inputs.items():
            inputs[k] = v.to(_kret_model.device)
    with torch.no_grad():
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=_local_tokenizer.eos_token_id,
        )
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs["attention_mask"]
        outputs = _kret_model.generate(**generate_kwargs)
    return _local_tokenizer.decode(outputs[0], skip_special_tokens=True)


# === REMOTE QUERY ===
def query_remote(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": REMOTE_MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# === UNIVERSAL INTERFACE ===


def has_internet(host="8.8.8.8", port=53, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def query_llm(prompt: str) -> str:
    use_local = USE_LOCAL and has_internet() is False
    if USE_LOCAL and has_internet():
        return query_local(prompt)
    else:
        return query_remote(prompt)
