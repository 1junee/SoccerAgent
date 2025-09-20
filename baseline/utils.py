
import base64
from PIL import Image
import json
import os
import csv
import subprocess
import re
from tqdm import tqdm
import cv2


def _normalize_option_token(token: str | None) -> str | None:
    if not token:
        return None
    token = token.strip()
    if not token:
        return None
    mapping = {'A': 'O1', 'B': 'O2', 'C': 'O3', 'D': 'O4'}
    first = token[0].upper()
    if first in mapping:
        return mapping[first]
    match = re.search(r'O([1-9])', token.upper())
    if match:
        return f"O{match.group(1)}"
    return None


def extract_answer_components(reply: str | None) -> tuple[str | None, str | None]:
    """Return (free_form_answer, option_choice) parsed from model reply.

    Works with both the legacy format that only returned an option and the
    updated prompt that includes labelled subjective/objective answers.
    """
    if not reply:
        return None, None

    text = reply.strip()
    if not text:
        return None, None

    # Capture labelled sections like "Subjective Answer:" and "Option:"
    free_form = None
    option = None

    labelled_free = re.search(r'(?:Subjective|Free)[^:]*:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    if labelled_free:
        free_form = labelled_free.group(1).strip()

    labelled_option = re.search(r'(?:Option|객관식)[^:]*:\s*([A-D]|O[1-9])', text, re.IGNORECASE)
    if labelled_option:
        option = _normalize_option_token(labelled_option.group(1))

    if option is None:
        option = _normalize_option_token(text)

    if free_form is None and option is not None:
        before_option = text.split(labelled_option.group(0))[0].strip() if labelled_option else text
        if before_option and before_option.upper() not in {option, option.replace('O', '')}:
            free_form = before_option

    if free_form:
        free_form = free_form.strip()
        if _normalize_option_token(free_form) == option:
            free_form = None

    return free_form or None, option


def extract_option(reply):
    _, option = extract_answer_components(reply)
    return option


def count_csv_rows(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count - 1


def ensure_csv_has_column(file_path: str, column_name: str) -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, mode='r', encoding='utf-8', newline='') as file:
        rows = list(csv.reader(file))
    if not rows:
        return
    header = rows[0]
    if column_name in header:
        return
    header.append(column_name)
    for row in rows[1:]:
        row.append('')
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def compress_video(input_path, output_path, crf=28):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = [
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", str(crf), output_path
    ]
    subprocess.run(command, check=True)
    return output_path


def compress_image(image_path, output_path, quality=50):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Image.open(image_path) as img:
        img.save(output_path, format="JPEG", quality=quality)
    return output_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    # print(f"Base64 image size: {sys.getsizeof(base64_image)} bytes")
    return base64_image
    

def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sort_files_by_number_in_name(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    stride = len(files) // 25
    files = files[::stride]
    return files


def videolist2imglist(video_path, num):
    base64_videos = []
    assign = []
    video_num = len(video_path)
    for i in range(video_num):
        assign.append(num / video_num)
    j = 0
    for video in video_path:
        video = cv2.VideoCapture(video)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        base64_images = []
        fps = assign[j]
        interval = int(frame_count / fps)
        for i in range(0, frame_count, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_images.append(base64.b64encode(buffer).decode("utf-8"))
        base64_videos.append(base64_images)
        j += 1
    return base64_videos
