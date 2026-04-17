"""Material Reasoner: 타겟 물체 조작 관점 물성 추론기 (GPT-4o Vision)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """너는 타겟 물체 조작 관점의 물성 추론기이다.
목표는 장면 전체가 아니라, 특정 task에서 타겟 물체를
"어떻게 다뤄야 하는지"를 판단하는 것이다.

지시사항:
1. 타겟 물체 하나만 분석하라.
2. 단순 재질 분류가 아니라, 조작 관점에서 물성을 해석하라.
3. 이 task를 수행할 때 어떤 제약이 발생하는지 추론하라.
4. tool 추천은 하지 말고, constraint만 출력하라.

반드시 추론할 것:
- material_hypotheses
- fragility
- slip_tendency
- deformability
- roll_risk

그리고 추가로:
- manipulation_difficulty
- required_interaction_type
- key_constraints

반드시 JSON만 출력하라. JSON 바깥의 설명문은 출력하지 마라.
반드시 모든 출력 텍스트는 한국어로 작성하라.

출력 JSON:
{
  "target_name": "...",
  "task": "...",
  "material_hypotheses": [
    {"material": "...", "probability": 0.0}
  ],
  "physical_properties": {
    "fragility": "low | medium | high",
    "slip_tendency": "low | medium | high",
    "deformability": "low | medium | high",
    "roll_risk": "none | low | medium | high"
  },
  "manipulation_analysis": {
    "difficulty": "easy | medium | hard",
    "required_interaction": "...",
    "constraints": ["...", "..."]
  },
  "uncertainty": ["..."]
}"""


def run_material_reasoner(
    image_path: Path,
    target_name: str,
    task: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    이미지 + 타겟 물체명 + task → 조작 제약 추론.

    Args:
        image_path: 장면 이미지 경로
        target_name: 타겟 물체 이름 (예: "business card", "명함")
        task: task 설명 (예: "좁은 틈 사이에 있는 명함을 꺼내라")
        api_key: OpenAI API key

    Returns:
        Material Reasoner output dict
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    image_b64 = _encode_image(image_path)

    user_message = (
        f"타겟 물체: {target_name}\n"
        f"task: {task}\n\n"
        "위 이미지와 정보를 바탕으로 타겟 물체의 조작 제약을 추론하라."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or ""
    usage = response.usage

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Material Reasoner: GPT 응답 JSON 파싱 실패: {e}\n"
            f"응답 미리보기: {raw_text[:300]}"
        )

    result["_meta"] = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }

    return result


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
