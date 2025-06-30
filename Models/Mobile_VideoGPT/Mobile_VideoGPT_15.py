import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .mobilevideogpt.utils import preprocess_input


def build_Mobile_VideoGPT_15():
    pretrained_path = "Your_Path/VRU_Accident/Models/Mobile_VideoGPT/Mobile_VideoGPT_15B" # Fill out your absolute path
    config = AutoConfig.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        torch_dtype=torch.float16
    ).to("cuda")

    return preprocess_input, tokenizer, model