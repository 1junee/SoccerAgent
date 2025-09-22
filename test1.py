from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# OpenRouter 클라이언트 초기화
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 간단한 채팅 테스트
resp = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-0324",   # 모델 이름
    messages=[
        {"role": "user", "content": "Hello, can you say hi in Korean?"}
    ]
)

print(resp.choices[0].message.content)
