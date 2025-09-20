from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = "DEEPSEEK_API_KEY"
if not api_key:
    raise RuntimeError("환경 변수 OPENROUTER_API_KEY 또는 DEEPSEEK_API_KEY가 설정되어 있지 않습니다.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key
)

completion = client.chat.completions.create(
  model="openai/gpt-4o",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life, one sentence?"
    }
  ],
  max_tokens=512
)

print(completion.choices[0].message.content)
