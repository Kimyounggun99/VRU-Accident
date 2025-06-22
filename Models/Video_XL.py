import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration



def build_LLaVA_NeXT_Video():
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to("cuda")

    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    return model, processor
