from datasets import load_dataset

ds = load_dataset("Homie0609/SoccerBench")  # 기본 split: "test"
print(ds)
# 각 항목의 "materials" 경로가 비디오/이미지 파일 위치를 가리킵니다.
