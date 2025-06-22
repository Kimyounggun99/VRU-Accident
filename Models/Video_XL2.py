from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
import torch


def build_Video_XL2():
         
    model_path = './Models/Video-XL-2'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device,quantization_config=None, attn_implementation="sdpa", torch_dtype=torch.float16, low_cpu_mem_usage=True)

    model.config.enable_sparse = False

    return model, tokenizer
    
