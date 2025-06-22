import torch
from transformers import AutoModelForCausalLM, AutoProcessor



def build_VideoLLaMA3():
    model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor
