import argparse
import os
import json
import av
import numpy as np
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from decord import VideoReader, cpu
import copy
import re
import torchvision.transforms as T

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import cv2

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help= 'VQA/Dense_Captioning')
    parser.add_argument('--dataset', type=str, required=True, help= 'CAP_DATA/DADA_2000/DoTA/MANUAL_DATA')
    parser.add_argument('--mode', type=str, required=True, help='generation/evaluation')     
    parser.add_argument('--save_path', type=str, help= './(Model_Response or results)/...')
    parser.add_argument('--load_path', type=str, help= './Model_Response/LLaVA_NeXT_Video_VQA_response.json')
    parser.add_argument('--model', type=str, required=True, help= 'LLaVA_NeXT_Video/')  
    parser.add_argument('--api_key', type=str, help= 'GPT_API_KEY/Gemini_API_Key')  
    return parser.parse_args()



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def InternVL_load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

###########

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def DataLoader(args):
    folder_path = os.path.join("./MetaData", args.dataset) #./MetaData

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    eval_data= {}
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for video, categories in tqdm(data.items(), desc=f"Loading {file_path}..."):

            eval_data[video] = {}

            if args.model =='LLaVA_NeXT_Video' or args.model =='LLaVA_OneVision' or 'Qwen' in args.model or args.model == 'Mobile_VideoGPT_15' or args.model == 'Video_XL2' or args.model == 'Video_XL_Pro' or args.model=='GPT_4o_mini':
                container = av.open(video)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)
            elif args.model == 'LLaVA_Video':
                clip,frame_time,video_time = load_video(video, 16, 1, force_sample=True)
            elif args.model=='Gemini_15_flash' or args.model == 'GPT_4o_mini':
                clip,frame_time,video_time = load_video(video, 8, 1, force_sample=True)
            elif 'InternVL' in args.model:
                
                clip, num_patches_list = InternVL_load_video(video, num_segments=8, max_num=1)
                clip = clip.to(torch.bfloat16).cuda()
                eval_data[video]['num_patches_list'] = num_patches_list
            eval_data[video]['annotation'] = categories
            eval_data[video]['frames'] = clip


    return eval_data


def extract_single_choice(text):

    candidates = re.findall(r'\b[A-D]\b', text)


    if candidates:
        return candidates[0]
    

    match = re.search(r'Answer:\s*\(?([A-D])\)?', text)
    if match:
        return match.group(1)

    return None

def build_conversation_template(args, annotation, video_path, all_items):
    question = annotation['question']
    options = annotation['options']
    
    if args.task == 'VQA':
        question_options= question + " " + options
    else:
        question_options = '''
        Provide a detailed description of this crash video. \n
        Use clear and complete sentences with appropriate traffic and crash-related terminology. \n
        Include descriptions of weather conditions, road type, and vehicle or pedestrian appearance (such as clothing and posture). \n
        Mention vehicle speed, trajectory, and movements, as well as any changes in the pedestrian's behavior. \n
        Focus on the dynamics of the collision, including vehicle approach, pedestrian movement, and final impact. \n
        '''
    if 'InternVL' in args.model:
        num_patches_list = all_items['num_patches_list']
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        conversation = video_prefix + question_options

    elif args.model == 'LLaVA_NeXT_Video' or args.model == 'LLaVA_OneVision' or 'Qwen' in args.model:
        
        conversation = [
            {
            
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question_options}"},
                    {"type": "video"},
                    ],
            },
        ]


    elif args.model == 'LLaVA_Video':
        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + "\n " + " " + question_options
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        conversation = conv.get_prompt()

    elif args.model == 'Video_XL2' or args.model == 'Mobile_VideoGPT_15':
        conversation = question_options
    elif args.model == 'Video_XL_Pro':
        conversation = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{question_options}<|im_end|>\n<|im_start|>assistant\n"
    elif args.model == 'Gemini_15_flash' or args.model == 'GPT_4o_mini':
        conversation = f"""
                        You are given a video and a question.
                        Please answer the question based on the video content.

                        Question: {question_options}
                        """

    return conversation


def postprocessor(args, output):

    if args.task == 'VQA':
        if args.model == 'LLaVA_NeXT_Video':
            final_output = output.split('ASSISTANT: ')[-1].split('.')[0]
        elif args.model == 'LLaVA_Video':
            final_output = output.split('.')[0]
        elif args.model == 'Video_XL2' or args.model == 'Video_XL_Pro' or args.model == 'GPT_4o_mini':
            final_output = output.split('.')[0]
        elif args.model == 'Gemini_15_flash':
            final_output = output
        elif args.model == 'LLaVA_OneVision':
            final_output = output.split('\n')[-1].split('.')[0]
        elif args.model == 'Mobile_VideoGPT_15':
            final_output = extract_single_choice(output)
        elif 'Qwen' in args.model:
            final_output = extract_single_choice(output.split('\n')[-1])
        elif 'InternVL' in args.model:
            final_output = extract_single_choice(output)
        else:
            breakpoint()
    else:

        if args.model == 'LLaVA_NeXT_Video':
            final_output = output.split('ASSISTANT: ')[-1]
        elif args.model == 'LLaVA_OneVision' or 'Qwen' in args.model:
            final_output = output.split('assistant\n')[-1]
        else:
            final_output = output

    return final_output


def save_prediction(args, save_dict):


    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)

    print(f"✅  Saved: {args.save_path}")