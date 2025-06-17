from langchain_community.llms import LlamaCpp
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import MODEL_PATH

def load_llm():
    import torch
     # GPU verification
    print("\n=== GPU Verification ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("Warning: No GPU detected - falling back to CPU")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=32,        # Mistral works well with more GPU layers 32/32 layers on GPU
        n_batch=256,
        n_ctx=32768,             # Mistral's context window
        max_tokens=1000,
        temperature=0.7,
        repeat_penalty=1.1,     # Helps with Mistral's response quality
        callback_manager=callback_manager,
        f16_kv=True,
        verbose=False
    )