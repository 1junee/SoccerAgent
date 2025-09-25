import sys, json, argparse
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline")
import os
import argparse
from multiagent_platform import EXECUTE_TOOL_CHAIN
from openai import OpenAI

import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from token_tracker import token_tracker

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

LOG_DIR = Path(PROJECT_PATH) / "result_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_log_paths(input_file: str | None) -> tuple[Path, Path]:
    """입력 파일명을 기반으로 오답/에러 로그 파일 경로를 생성합니다."""
    stem = Path(input_file).stem if input_file else "default"
    incorrect_path = LOG_DIR / f"{stem}_incorrect.log"
    error_path = LOG_DIR / f"{stem}_errors.log"
    return incorrect_path, error_path


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


def _truncate(text: Optional[str], limit: int = 200) -> Optional[str]:
    if text is None:
        return None
    return text[:limit] + "..." if len(text) > limit else text


def append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

INSTRUCTION = f"""
You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
Before I have used a helpful soccer multi-agent system to solve this process, I will tell you the total process of how agent deal with this problem. 
Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4'). 
Do not include any other text or explanations!!!
"""

def workflow(input_text, Instruction=INSTRUCTION, follow_up_prompt=None, max_tokens_followup=1500):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
    )
    first_round_reply = completion.choices[0].message.content
    return first_round_reply

import re

def process_football_question(input_dict, question_index=None):
    if "openA_process" in input_dict and "answer" in input_dict:
        return input_dict

    question = input_dict.get("Q", "")
    materials = input_dict.get("materials", "")

    options = {key: value for key, value in input_dict.items() if key.startswith("O")}
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

    # 쿼리 정보 추가 (추적용)
    query_info = {
        "question_index": question_index,
        "question": question[:100] + "..." if len(question) > 100 else question,
        "has_materials": bool(materials)
    }

    openA_process = EXECUTE_TOOL_CHAIN(question, materials, options_str)

    prompt = f"""
This football question is "{question}". The four corresponding options are:
{options_str}

The processing through the multi-agent platform is as follows:
{openA_process}

Please provide your answer:
"""

    processed_prompt = workflow(prompt)

    answer_match = re.search(r"O\d+", processed_prompt)
    answer = answer_match.group(0) if answer_match else None

    result_dict = input_dict.copy()
    result_dict["openA_process"] = openA_process
    result_dict["answer"] = answer

    return result_dict


from tqdm import tqdm
import json

def process_json_file(input_file, output_file, incorrect_log_path: Path, error_log_path: Path):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        progress_bar = tqdm(data_list, desc="Processing (Accuracy: N/A)", unit="item")

        incorrect_logs: list[dict] = []
        error_logs: list[dict] = []

        for i, item in enumerate(progress_bar):
            try:
                updated_item = process_football_question(item, i)
                data_list[i] = updated_item

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=4)

                expected = updated_item.get("closeA")
                predicted = updated_item.get("answer")
                if expected is not None and predicted != expected:
                    incorrect_logs.append(
                        {
                            "timestamp": _timestamp(),
                            "mode": "full_run",
                            "index": i,
                            "question": _truncate(updated_item.get("Q"), 200),
                            "expected": expected,
                            "predicted": predicted,
                            "has_materials": bool(updated_item.get("materials")),
                            "output_file": output_file,
                            "openA_process_preview": _truncate(updated_item.get("openA_process"), 400),
                        }
                    )

            except ValueError as ve:
                print(f"ValueError processing item {i}: {ve}")
                error_logs.append(
                    {
                        "timestamp": _timestamp(),
                        "mode": "full_run",
                        "index": i,
                        "question": _truncate(item.get("Q"), 200),
                        "error_type": type(ve).__name__,
                        "message": str(ve),
                        "output_file": output_file,
                    }
                )
                continue

            except Exception as e:
                print(f"Unexpected error processing item {i}: {e}")
                error_logs.append(
                    {
                        "timestamp": _timestamp(),
                        "mode": "full_run",
                        "index": i,
                        "question": _truncate(item.get("Q"), 200),
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "output_file": output_file,
                    }
                )
                continue

            correct_count = 0
            total_count = 0
            for entry in data_list:
                if "openA_process" in entry and "answer" in entry:
                    total_count += 1
                    if entry["answer"] == entry.get("closeA"):
                        correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0

            progress_bar.set_description(f"Processing (Accuracy: {accuracy:.2%})")
            progress_bar.refresh() 

        append_jsonl(incorrect_log_path, incorrect_logs)
        append_jsonl(error_log_path, error_logs)

        print(f"Processing completed. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")

# 1% 샘플이 이미 준비되어 있다고 가정하는 비용 추정 함수
def process_sample_for_cost_estimation(
    input_file,
    output_file,
    sample_ratio=0.01,
    incorrect_log_path: Path | None = None,
    error_log_path: Path | None = None,
):
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if sample_ratio <= 0:
        raise ValueError("--sample_ratio 값은 0보다 커야 합니다.")

    # 입력 파일이 이미 샘플 데이터라고 가정하므로 추가 샘플링을 수행하지 않습니다.
    sample_data = full_data
    sample_size = len(sample_data)
    estimated_total_size = max(sample_size, int(round(sample_size / sample_ratio)))

    print(
        "Processing {sample_size} pre-sampled items (assumed {ratio:.0%} of ≈{total} total) for cost estimation...".format(
            sample_size=sample_size,
            ratio=sample_ratio,
            total=estimated_total_size,
        )
    )

    # 샘플 처리
    incorrect_logs: list[dict] = []
    error_logs: list[dict] = []
    for i, item in enumerate(sample_data):
        try:
            updated_item = process_football_question(item, i)
            sample_data[i] = updated_item

            expected = updated_item.get("closeA")
            predicted = updated_item.get("answer")
            if expected is not None and predicted != expected:
                incorrect_logs.append(
                    {
                        "timestamp": _timestamp(),
                        "mode": "sample_mode",
                        "index": i,
                        "question": _truncate(updated_item.get("Q"), 200),
                        "expected": expected,
                        "predicted": predicted,
                        "has_materials": bool(updated_item.get("materials")),
                        "output_file": output_file,
                        "openA_process_preview": _truncate(updated_item.get("openA_process"), 400),
                    }
                )

            # 5개마다 중간 통계 출력
            if (i + 1) % 5 == 0:
                print(f"\n--- After {i+1} samples ---")
                token_tracker.print_statistics()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            error_logs.append(
                {
                    "timestamp": _timestamp(),
                    "mode": "sample_mode",
                    "index": i,
                    "question": _truncate(item.get("Q"), 200),
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "output_file": output_file,
                }
            )
            continue

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

    append_jsonl(incorrect_log_path or LOG_DIR / "sample_incorrect.log", incorrect_logs)
    append_jsonl(error_log_path or LOG_DIR / "sample_errors.log", error_logs)

    # 최종 비용 추정
    estimated_cost = token_tracker.estimate_full_cost(sample_size, estimated_total_size)

    return estimated_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file containing football questions.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input JSON file. See https://huggingface.co/datasets/Homie0609/SoccerBench/raw/main/qa/q1.json as an example",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output JSON file. You can just set an json path.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run in sampling mode for cost estimation",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.01,
        help="비용 추정을 위해 입력 샘플이 전체 데이터에서 차지하는 비율 (기본값: 0.01)",
    )

    args = parser.parse_args()

    incorrect_log_path, error_log_path = resolve_log_paths(args.input_file)

    if args.sample_mode:
        estimated_cost = process_sample_for_cost_estimation(
            args.input_file,
            args.output_file,
            args.sample_ratio,
            incorrect_log_path=incorrect_log_path,
            error_log_path=error_log_path,
        )
        print(f"\nFinal estimated cost for full dataset: ${estimated_cost:.2f}")
    else:
        process_json_file(args.input_file, args.output_file, incorrect_log_path, error_log_path)
