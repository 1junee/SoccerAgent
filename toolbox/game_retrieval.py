#### GAME INFO RETRIEVAL & MATCH HISTORY RETRIEVAL

import json
import json
from tqdm import tqdm
import argparse, os
from functools import lru_cache
from typing import List, Dict

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

######################## Parameters ########################
PROJECT_PATH = "/home/work/wonjun/study/agent/SoccerAgent"

# Use local Qwen2.5-VL-7B-Instruct for text generation
QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@lru_cache(maxsize=1)
def _load_qwen_vl(model_id: str = QWEN_VL_MODEL_ID):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return processor, model


def _qwen_chat(messages: List[Dict], max_new_tokens: int = 512) -> str:
    processor, model = _load_qwen_vl()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


def workflow(input_text, Instruction, follow_up_prompt=None, max_tokens_followup=1500):
    # First turn
    messages = [
        {"role": "system", "content": Instruction},
        {"role": "user", "content": input_text},
    ]
    first_round_reply = _qwen_chat(messages, max_new_tokens=512)

    if follow_up_prompt:
        messages = [
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": first_round_reply},
            {"role": "user", "content": follow_up_prompt},
        ]
        second_round_reply = _qwen_chat(messages, max_new_tokens=max_tokens_followup)
        return first_round_reply, second_round_reply
    else:
        return first_round_reply



def generate_commentary_from_json_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    event_list = data.get("annotations", [])
    if not event_list:
        return "No annotations found in the JSON file."
    
    result = []
    
    for event in reversed(event_list): 
        timestamp = event.get("contrastive_aligned_gameTime", "")
        if not timestamp:
            timestamp = event.get("gameTime", "")
        if not timestamp:
            continue  
        
        try:
            half, time = timestamp.split(" - ")
            if half == "1":
                half_str = "1st half"
            elif half == "2":
                half_str = "2nd half"
            else:
                continue 
        except ValueError:
            continue 
        
        description = event.get("description", "")
        if not description:
            continue
        
        commentary_line = f"{half_str} - {time} \"{description}\""
        result.append(commentary_line)
    
    return "\n".join(result)

def generate_commentary_from_json_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    comments_list = data.get("comments", [])
    if not comments_list:
        return "No comments found in the JSON file."
    
    result = []
    
    for comment in comments_list:
        half = comment.get("half")
        if half not in [1, 2]:
            continue 
        
        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue 
        
        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue 
        
        half_str = "1st half" if half == 1 else "2nd half"
        
        commentary_line = f"{half_str} - {timestamp} \"{comments_text}\""
        result.append(commentary_line)
    
    return "\n".join(result)

def get_match_info_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "annotations" in data:
        del data["annotations"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

def get_match_info_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "comments" in data:
        del data["comments"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

def get_match_info(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return get_match_info_matchtime(json_file_path)
    else:
        return get_match_info_1988(json_file_path)
    
def generate_commentary_from_json(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return generate_commentary_from_json_matchtime(json_file_path)
    else:
        return generate_commentary_from_json_1988(json_file_path)
    
def MATCH_HISTORY_RETRIEVAL(query, material):
    if len(material) == 0:
        return "Something went wrong with this question."
    
    file_path = os.path.join(PROJECT_PATH, material[0])
    match_history = generate_commentary_from_json(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match history information has been found as following shows, you need to answer the question based on the information provided:

    {match_history}

Please provide the answer based on the match history information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer

def GAME_INFO_RETRIEVAL(query, material):
    if len(material) == 0:
        return "Something went wrong with this question."
    
    file_path = os.path.join(PROJECT_PATH, material[0])
    match_info = get_match_info(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match related information has been found as following shows, you need to answer the question based on the information provided:

    {match_info}

Please provide the answer based on the match related information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer
