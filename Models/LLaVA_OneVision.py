import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def build_LLaVA_OneVision():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to("cuda")
    
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor
