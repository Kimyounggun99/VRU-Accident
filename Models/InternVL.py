import torch
from transformers import AutoTokenizer, AutoModel


def build_InternVL(args):
    

    if args.model == 'InternVL2_8B':
        path = "OpenGVLab/InternVL2-8B"
    
    elif args.model == 'InternVL25_1B':
        path = "OpenGVLab/InternVL2_5-1B"
    elif args.model == 'InternVL25_2B':
        path = "OpenGVLab/InternVL2_5-2B"
    elif args.model == 'InternVL25_4B':
        path = "OpenGVLab/InternVL2_5-4B"
    elif args.model == 'InternVL25_8B':
        path = "OpenGVLab/InternVL2_5-8B"

    elif args.model == 'InternVL3_2B':
        path = "OpenGVLab/InternVL3-2B"
    elif args.model == 'InternVL3_8B':
        path = "OpenGVLab/InternVL3-8B"

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
    return model, tokenizer