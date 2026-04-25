import json
import base64
import numpy as np
<<<<<<< HEAD
from dotenv import load_dotenv

load_dotenv()
client = os.getenv("OPENAI_API_KEY")  # OPENAI_API_KEY 환경변수 자동 참조
=======
from openai import OpenAI

client = OpenAI(api_key="")  # OPENAI_API_KEY 환경변수 자동 참조
>>>>>>> origin/subin/module2c-3-pipeline

# ── 유틸: numpy RGB 이미지 → base64 PNG 문자열 ────────────────────────────────
def _rgb_to_base64(rgb: np.ndarray) -> str:
    """
    (H, W, 3) uint8 numpy 배열 → base64 인코딩된 PNG 문자열.
    PIL 없이 cv2로 변환 (cv2가 없으면 PIL fallback).
    """
    try:
        import cv2
        _, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except ImportError:
        from PIL import Image
        import io
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_scene_prompt(scene_info: dict) -> str:
    """
    수집된 씬 정보를 GPT에게 전달할 자연어 프롬프트로 변환.
    """
    lines = ["아래는 PyBullet 시뮬레이션 tabletop 위 물체들의 상태입니다.\n"]
    
    for obj in scene_info["objects"]:
        pos = obj["position"]
        ori = obj["orientation_euler_deg"]
        size = obj["size_aabb_m"]
        
        lines.append(
            f"- 물체명: {obj['label']}\n"
            f"  위치(m): x={pos['x']}, y={pos['y']}, z={pos['z']}\n"
            f"  방향(deg): roll={ori['roll']}, pitch={ori['pitch']}, yaw={ori['yaw']}\n"
            f"  크기(AABB, m): width={size['width']}, depth={size['depth']}, height={size['height']}\n"
        )
    
    lines.append(
        "\n위 정보를 바탕으로 다음을 수행해주세요:\n"
        "1. 각 물체의 종류와 현재 상태(세워있음/뒤집힘/기울어짐 등)를 분류해주세요.\n"
        "2. 로봇팔이 집기 쉬운 물체부터 어려운 물체 순으로 우선순위를 매겨주세요.\n"
        "3. 각 물체에 대해 간략한 설명을 한 문장으로 작성해주세요.\n"
        "JSON 형식으로 응답해주세요."
    )
    
    return "\n".join(lines)


def query_gpt_about_scene(scene_info: dict, model: str = "gpt-4o") -> dict:
    
    prompt = build_scene_prompt(scene_info)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 로봇 조작 연구를 돕는 AI입니다. "
                    "tabletop 위 물체들의 위치·방향·크기 데이터를 분석하여 "
                    "로봇팔 그래스핑 우선순위와 물체 상태를 JSON으로 반환합니다."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # 결정론적 응답 선호
        response_format={"type": "json_object"},  # GPT-4o JSON 모드
    )
    
    raw_text = response.choices[0].message.content
    
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None
    
    return {
        "prompt": prompt,
        "raw_response": raw_text,
        "parsed": parsed,
    }

# ── Vision + 텍스트로 타겟 물체 선택 ────────────────────────────────────
def query_gpt_pick_target(scene_info: dict, task: str = "테이블 위에서 가장 집기 쉬운 물체를 하나 골라주세요.", rgb_image: np.ndarray = None, model: str = "gpt-4o") -> dict:
    """
    GPT Vision (이미지 + 텍스트)으로 타겟 물체를 선택.
 
    rgb_image가 주어지면 GPT-4o Vision으로 이미지를 함께 전송.
    rgb_image가 None이면 기존 텍스트 전용 방식으로 fallback.
 
    Returns:
        {
            "target_label":   "mug",
            "grasp_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "reason":         "선택 이유 한 문장"
        }
    """
    # 씬 정보를 텍스트로 변환
    lines = [f"작업 지시: {task}\n", "현재 테이블 위 물체 목록:\n"]
    for obj in scene_info["objects"]:
        pos  = obj["position"]
        ori  = obj["orientation_euler_deg"]
        size = obj["size_aabb_m"]
        lines.append(
            f"- 물체명: {obj['label']}\n"
            f"  위치(m): x={pos['x']}, y={pos['y']}, z={pos['z']}\n"
            f"  방향(deg): roll={ori['roll']}, pitch={ori['pitch']}, yaw={ori['yaw']}\n"
            f"  크기(m): width={size['width']}, depth={size['depth']}, height={size['height']}\n"
        )
    lines.append(
        "\n반드시 아래 JSON 형식으로만 응답하세요:\n"
        "{\n"
        '  "target_label": "물체명",\n'
        '  "grasp_position": {"x": 0.0, "y": 0.0, "z": 0.0},\n'
        '  "reason": "선택 이유 한 문장"\n'
        "}"
    )
    text_prompt = "\n".join(lines)

# ── 메시지 구성: Vision 여부에 따라 분기 ──────────────────────────────────
    if rgb_image is not None:
        b64 = _rgb_to_base64(rgb_image)
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",   # high: 세밀한 물체 인식
                },
            },
            {
                "type": "text",
                "text": (
                    "위 이미지는 PyBullet 가상 카메라로 촬영한 tabletop 씬입니다.\n"
                    "이미지에서 각 물체의 시각적 특성(형태, 크기, 기울기, 주변 공간)을 "
                    "아래 수치 정보와 함께 고려하여 판단하세요.\n\n"
                    + text_prompt
                ),
            },
        ]
        print("[GPT Vision] 이미지 + 텍스트로 타겟 선택 요청 중...")
    else:
        # fallback: 텍스트 전용
        user_content = text_prompt
        print("[GPT] 텍스트 전용으로 타겟 선택 요청 중...")
 
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 로봇팔 조작을 제어하는 AI입니다. "
                    "tabletop 씬 이미지와 물체 수치 정보를 함께 분석하여 "
                    "로봇팔이 집기 가장 적합한 물체를 JSON으로 반환합니다. "
                    "target_label은 반드시 제공된 물체 목록의 label과 정확히 일치해야 합니다."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=512,
    )
 
    raw_text = response.choices[0].message.content
    parsed = json.loads(raw_text)
    print(f"[GPT 결정] 타겟: {parsed.get('target_label')} / 이유: {parsed.get('reason')}")
    return parsed

# ── (선택) 멀티뷰: 여러 카메라 각도 이미지를 한 번에 전송 ────────────────────
def query_gpt_pick_target_multiview(
    scene_info: dict,
    task: str,
    rgb_images: list,           # [(H,W,3), ...] — 여러 시점 이미지 리스트
    view_labels: list = None,   # ["front", "top", "side"] 등 선택적 레이블
    model: str = "gpt-4o",
) -> dict:
    """
    여러 카메라 시점 이미지를 동시에 GPT Vision에 전송.
    씬이 복잡하거나 물체가 가려진 경우에 유용.
    """
    if view_labels is None:
        view_labels = [f"view_{i+1}" for i in range(len(rgb_images))]
 
    lines = [f"작업 지시: {task}\n", "현재 테이블 위 물체 목록:\n"]
    for obj in scene_info["objects"]:
        pos  = obj["position"]
        size = obj["size_aabb_m"]
        lines.append(
            f"- {obj['label']}: 위치({pos['x']}, {pos['y']}, {pos['z']}), "
            f"크기({size['width']}×{size['depth']}×{size['height']}m)\n"
        )
    lines.append(
        "\n반드시 아래 JSON 형식으로만 응답하세요:\n"
        '{"target_label": "물체명", "grasp_position": {"x":0,"y":0,"z":0}, "reason": "이유"}'
    )
    text_prompt = "\n".join(lines)
 
    # 이미지들을 순서대로 content에 삽입
    user_content = []
    for img, label in zip(rgb_images, view_labels):
        b64 = _rgb_to_base64(img)
        user_content.append({
            "type": "text",
            "text": f"[{label} 시점]"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    user_content.append({"type": "text", "text": text_prompt})
 
    print(f"[GPT Vision Multiview] {len(rgb_images)}개 시점 이미지 전송 중...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 로봇팔 조작을 제어하는 AI입니다. "
                    "여러 시점의 tabletop 씬 이미지와 수치 정보를 분석하여 "
                    "최적의 파지 대상을 JSON으로 반환합니다."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=512,
    )
 
    raw_text = response.choices[0].message.content
    parsed = json.loads(raw_text)
    print(f"[GPT 결정] 타겟: {parsed.get('target_label')} / 이유: {parsed.get('reason')}")
    return parsed