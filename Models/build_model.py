from .LLaVA_NeXT_Video import build_LLaVA_NeXT_Video
from .LLaVA_Video import build_LLaVA_Video
from .Video_XL2 import build_Video_XL2
from .Video_XL_Pro import build_Video_XL_Pro
from .LLaVA_OneVision import build_LLaVA_OneVision
from .Mobile_VideoGPT.Mobile_VideoGPT_15 import build_Mobile_VideoGPT_15
from .Qwen import build_Qwen
from .InternVL import build_InternVL

def build_model(args):

    if args.model == 'LLaVA_NeXT_Video':
        model, processor = build_LLaVA_NeXT_Video()
        return model, processor
    elif args.model == 'LLaVA_Video':
        image_processor, tokenizer, tokenizer_image_token, model = build_LLaVA_Video()
        return image_processor, tokenizer, tokenizer_image_token, model
    elif args.model == 'Video_XL2':
        model, tokenizer = build_Video_XL2()
        return model, tokenizer
    elif args.model == 'Video_XL_Pro':
        model, tokenizer = build_Video_XL_Pro()
        return model, tokenizer
    elif args.model == 'LLaVA_OneVision':
        model, processor = build_LLaVA_OneVision()
        return model, processor
    elif args.model == 'Mobile_VideoGPT_15':
        image_processor, tokenizer, model = build_Mobile_VideoGPT_15()
        return image_processor, tokenizer, model
    elif 'Qwen' in args.model: 
        model, processor = build_Qwen(args)
        return model, processor
    elif 'InternVL' in args.model:
        model, tokenizer = build_InternVL(args)
        return model, tokenizer
     
    else:
        breakpoint()

    