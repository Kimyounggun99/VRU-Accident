
import json
import re
import string
from collections import defaultdict
import os 
import torch
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX
from utils.utils import get_arg_parse, DataLoader, build_conversation_template, postprocessor, save_prediction
from utils.evaluation import VQA_eval, Dense_Captioning_eval
from Models.build_model import build_model
from Models.Gemini import Gemini_15_flash
from Models.GPT import GPT_4o_mini
import google.generativeai as genai

import sys
sys.path.append("./Your_Path/VRU_Accident/Models/Video_XL_Pro_3B") # Fill out your path
from videoxlpro.videoxlpro.demo_utils import process_video, load_image_processor, generate_response 

from tqdm import tqdm



def main(args):
    ############## Load model #####################
    if args.mode == 'generation':
        data_dict= DataLoader(args)
        if args.model == 'LLaVA_NeXT_Video' or args.model == 'LLaVA_OneVision' or 'Qwen' in args.model:
            model, processor = build_model(args)
        elif args.model == 'LLaVA_Video':
            image_processor, tokenizer, tokenizer_image_token, model = build_model(args)
            model.eval()
        elif args.model == 'Video_XL2' or args.model == 'Video_XL_Pro' or 'InternVL' in args.model:
            model, tokenizer = build_model(args)
        elif args.model == 'Mobile_VideoGPT_15':
            image_processor, tokenizer, model = build_model(args)


        save_dict = {}
 
        for video_path, all_items in tqdm(data_dict.items(), desc= f'{args.model} Sentence Generation...'):

            save_dict[video_path] = {}

            video_frames= all_items['frames']

            #### Video Processor
            if args.model =='LLaVA_Video':
                video_frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(dtype=torch.bfloat16, device=model.device)
                video_frames = [video_frames]

            elif args.model == 'Video_XL_Pro':
                    image_processor = load_image_processor(model, tokenizer)
                    video_tensor,time_embed = process_video(video_path,tokenizer, image_processor, model.device, 16)

            #### Model Response Generation
            for category, annotation in all_items['annotation'].items():

                if args.task == 'VQA':
                    save_dict[video_path][category] = {}
                
                conversation = build_conversation_template(args, annotation, video_path, all_items)

                gt= annotation['GT']

                if args.task == 'VQA':
                    max_new_tokens = 10
                elif args.task == 'Dense_Captioning':
                    max_new_tokens = 200
                    
                else:
                    ValueError('We only support VQA and Dense_Captioning tasks.')

                if args.model == 'LLaVA_NeXT_Video' or args.model == 'LLaVA_OneVision' or 'Qwen' in args.model:
                    try:
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs_video = processor(text=prompt, videos=video_frames, padding=True, return_tensors="pt").to(model.device)
                        output = model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=False)
                        response = processor.decode(output[0][2:], skip_special_tokens=True)
                    except:
                        reponse = None
                        print("CUDA OOM!! \n")
                elif args.model == 'LLaVA_Video':
                    input_ids = tokenizer_image_token(conversation, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
                    with torch.no_grad():
                        cont = model.generate(input_ids, images=video_frames, modalities= ["video"], do_sample=False, max_new_tokens=max_new_tokens,)
                    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                    
                elif args.model == 'Video_XL2':
                    with torch.inference_mode():
                        response = model.chat(video_path, tokenizer, conversation, chat_history=None, return_history=False,max_num_frames=16, sample_fps=1, max_sample_fps=4, generation_config={ "do_sample": False, "temperature": 0.01, "top_p": 0.001, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens})
                
                elif args.model == 'Video_XL_Pro':
                    response = generate_response(model, tokenizer, conversation, video_tensor,time_embed, {"do_sample": True, "temperature": 0.01, "top_p": 0.001, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens})

                elif args.model == 'Gemini_15_flash':   
                    response = Gemini_15_flash(video_frames, conversation, args.api_key)
                elif args.model == 'GPT_4o_mini':
                    response = GPT_4o_mini(video_frames, conversation, args.api_key)

                elif args.model == 'Mobile_VideoGPT_15':
                    input_ids, video_frames, context_frames, stop_str = image_processor(
                        model, tokenizer, video_path, conversation
                    )
                    with torch.inference_mode():
                        output_ids = model.generate(input_ids,images=torch.stack(video_frames, dim=0).half().cuda(),context_images=torch.stack(context_frames, dim=0).half().cuda(),do_sample=False,temperature=0,top_p=1,num_beams=1,max_new_tokens=max_new_tokens,use_cache=True,)
                        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    if response.endswith(stop_str):
                        response = response[:-len(stop_str)].strip()

                elif 'InternVL' in args.model:
                    response, history = model.chat(tokenizer, video_frames, conversation, {'max_new_tokens':max_new_tokens, 'do_sample':True},
                               num_patches_list=all_items['num_patches_list'], history=None, return_history=True)

                final_output = postprocessor(args, response)

                if args.task == 'VQA':
                    save_dict[video_path][category]['pred'] = final_output
                    save_dict[video_path][category]['GT'] = gt
                else:
                    save_dict[video_path]['pred'] = final_output
                    break

        save_prediction(args, save_dict)  

    else:
        if args.task == 'VQA':
            VQA_eval(args)
        elif args.task == 'Dense_Captioning':
            Dense_Captioning_eval(args)








if __name__ == '__main__':
    args= get_arg_parse()
    main(args)



