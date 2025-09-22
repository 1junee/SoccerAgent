"""CloseQA tool: map an open-ended answer onto a multiple-choice option."""

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_OPENAI_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _OPENAI_CLIENT


_INSTRUCTION = (
    "You are an assistant that maps an already-generated open-form answer to the most "
    "appropriate option among multiple-choice candidates. Respond with the option ID "
    "only (e.g., O1, O2)."
)


def CLOSE_QA(query: str, material):
    """Select the best option label for the provided open-ended answer.

    Args:
        query: String containing the question, the list of options, and the open-form answer.
        material: Unused; kept for tool interface compatibility.
    Returns:
        The selected option label or an error string on failure.
    """

    client = _get_client()
    prompt = (
        "Question, options, and open-form answer:\n"
        f"{query}\n\nRespond ONLY with the option ID (e.g., O1)."
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            max_tokens=16,
        )
    except Exception as err:  # pragma: no cover - defensive path
        return f"CloseQA failed: {err}"

    response = completion.choices[0].message.content.strip()
    # Ensure we only return the first token resembling an option ID.
    return response.split()[0] if response else ""

