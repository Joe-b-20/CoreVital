# CoreVital inference backends (Hugging Face, vLLM, llama.cpp, TGI).
# Use Backend interface for backend-agnostic instrumentation.

from CoreVital.backends.base import Backend, BackendCapabilities
from CoreVital.backends.huggingface import HuggingFaceBackend
from CoreVital.backends.llama_cpp_backend import LlamaCppBackend
from CoreVital.backends.tgi_backend import TGIBackend
from CoreVital.backends.vllm_backend import VLLMBackend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "HuggingFaceBackend",
    "LlamaCppBackend",
    "TGIBackend",
    "VLLMBackend",
]
