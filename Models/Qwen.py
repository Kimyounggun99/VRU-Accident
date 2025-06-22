from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration
def build_Qwen(args):

    if args.model == 'Qwen25_VL':
        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype= torch.float16, low_cpu_mem_usage=True, 
        ).to('cuda')
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    elif args.model == 'Qwen2_VL_2B':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype= torch.float16, low_cpu_mem_usage=True, 
        ).to('cuda')
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    elif args.model == 'Qwen2_VL_7B':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype= torch.float16, low_cpu_mem_usage=True, 
        ).to('cuda')
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor
