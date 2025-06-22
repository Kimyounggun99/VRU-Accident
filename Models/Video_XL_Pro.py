import torch
import transformers
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings



def build_Video_XL_Pro():
         
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings('ignore')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "./Models/Video_XL_Pro_3B"


    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    return model, tokenizer

