import sys
import torch
from diffusers import AutoPipelineForText2Image

print("Starting Hugging Face Model Load test...")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

model_id = "stabilityai/sd-turbo"
try:
    print(f"Attempting to load pipeline for {model_id}...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True,
        local_files_only=True
    )
    print("Pipeline loaded successfully!")
    pipe.to("mps")
    print("Pipeline moved to MPS successfully!")
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
