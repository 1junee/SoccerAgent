import json, os
from tqdm import tqdm
import argparse
from functools import lru_cache
from typing import List, Dict

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from project_path import PROJECT_PATH

######################## Parameters ########################

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


def generate_textual_RAG_prompt(question, textual_material):
    """
    Generates a prompt based on the question and textual material.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated prompt.
    """
    # Check if textual_material is a JSON file path
    if isinstance(textual_material, str) and textual_material.endswith(".json"):
        try:
            # Read the JSON file and convert it to a formatted string
            with open(textual_material, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
                textual_material = formatted_json
        except Exception as e:
            return f"Failed to read JSON file: {e}"

    # Generate the prompt
    prompt = f"""
    Question: {question}
    Contextual Material: {textual_material}
    Please answer the question based on the provided contextual material.
    """
    return prompt

def TEXTUAL_RETRIEVAL_AUGMENT(question, textual_material):
    """
    Retrieves and augments textual material to answer the question using the workflow function.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated answer.
    """
    # Define the instruction for the agent
    Instruction = "You are an assistant that answers questions based on provided contextual material."
    file_path = os.path.join(PROJECT_PATH, textual_material[0])

    # Generate the prompt
    prompt = generate_textual_RAG_prompt(question, file_path)
    answer = None
    try:
    # Call the workflow function to generate the answer
        answer = workflow(prompt, Instruction)
    except:
        answer = "Failed in LLM Generation."
    # Return the QA output
    return answer
