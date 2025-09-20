# API 키 안전 적용 가이드 (SoccerAgent)

## 1. 환경 변수 사용
- `platform_full_version.py`, `multiagent_platform.py`, `baseline/model.py` 등에서 하드코딩된 API 키를 제거하고 `os.environ.get()`으로 받아옵니다.
- 키가 설정되지 않았을 때는 `RuntimeError` 등으로 명확히 실패하도록 처리합니다.

```python
import os
from openai import OpenAI

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("환경 변수 DEEPSEEK_API_KEY 가 설정되어 있지 않습니다.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
```

## 2. .env 파일로 로컬 개발 환경 구성
1. 프로젝트 루트에 `.env` 파일을 만들고 아래와 같이 키를 저장합니다.
   ```dotenv
   DEEPSEEK_API_KEY=여기에_딥시크_API_키
   GPT_API_KEY=여기에_OpenAI_API_키
   ```
2. `.env` 파일은 반드시 `.gitignore`에 포함해 Git에 올라가지 않게 합니다.
3. Python에서는 `python-dotenv` 패키지를 사용하여 개발 환경에서만 `.env`를 로드합니다.

```python
from dotenv import load_dotenv
load_dotenv()
```

## 3. 배포 환경에서의 보안 주의사항
- CI/CD, Docker, 서버 환경에서는 환경 변수나 시크릿 매니저(예: GitHub Actions Secrets, AWS Secrets Manager)를 통해 키를 주입합니다.
- 로그나 예외 메시지에 API 키가 노출되지 않도록 주의합니다.
- 키가 노출되었다면 즉시 폐기하고 새 키를 발급받습니다.

## 4. 코드 구조 제안
- API 키가 필요한 함수나 클래스에 키를 직접 넘기기보다, 초기화 시점에만 한 번 불러오도록 구성합니다.
- 여러 서비스를 사용할 경우 `DEEPSEEK_API_KEY`, `GPT_API_KEY` 등 명확한 이름으로 구분합니다.

## 5. 점검 체크리스트
- [ ] 하드코딩된 키가 있는지 확인
- [ ] `.env`가 버전 관리에서 제외되었는지 확인
- [ ] 로컬/운영 환경에서 각각 올바른 방식으로 키가 주입되는지 확인
- [ ] 예외/로그에 키가 노출되지 않는지 확인
