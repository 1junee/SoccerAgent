from huggingface_hub import snapshot_download

# 환경변수(선택): 재시도 횟수와 타임아웃 늘리기
# export HF_HUB_MAX_RETRIES=20
# export HF_HUB_TIMEOUT=90

snapshot_download(
    repo_id="Homie0609/SoccerBench",
    repo_type="dataset",
    local_dir="./datasets/SoccerBench",
    local_dir_use_symlinks=False,
    max_workers=2,              # 동시성 낮추기(핵심)
    resume_download=True,       # 끊겨도 이어받기
)

snapshot_download(
    repo_id="SJTU-AI4Sports/SoccerWiki",
    repo_type="dataset",
    local_dir="./datasets/SoccerWiki",
    local_dir_use_symlinks=False,
    max_workers=2,
    resume_download=True,
)
