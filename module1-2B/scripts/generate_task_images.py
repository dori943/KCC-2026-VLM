"""Generate task scene images via DALL-E 3.

기존 cases/images/*.png 를 새 위급 시나리오 이미지로 덮어쓴다.
원본은 cases/images/_backup/ 에 자동 백업.

사용:
    export OPENAI_API_KEY="sk-..."
    python scripts/generate_task_images.py            # 5장 모두
    python scripts/generate_task_images.py task2      # 특정 task만
"""
from __future__ import annotations

import base64
import os
import shutil
import sys
from pathlib import Path

from openai import OpenAI


PROMPTS = {
    "card_from_gap": (
        "A real DSLR photograph (not illustration, not cartoon, not 3D render). "
        "Indoor modern living room. A folded white prescription paper is "
        "DEEPLY WEDGED under a beige fabric sofa, with only a small corner "
        "barely peeking out from the very narrow gap between the sofa bottom "
        "and the wooden floor. The gap is only about 2-3 cm tall — clearly "
        "too narrow for a hand. Eye-level shot from floor angle, sharp focus, "
        "soft natural daylight. No people, no animals. Photorealistic."
    ),
    "deep_hole_reach": (
        "A real photograph (not illustration, not cartoon). Top-down view "
        "looking THROUGH a metal sewer drain grate set into a city sidewalk. "
        "The viewer sees DEEP DOWN into a dark drain hole below the grate, "
        "and at the bottom of the drain, far below, a metallic silver car "
        "key with black plastic head is visible in shadow. Strong sense of "
        "depth — the key is clearly unreachable by hand. Daylight, urban street "
        "context. Sharp focus. Photorealistic. No people."
    ),
    "suspended_target": (
        "A real photograph (not illustration, not cartoon, not animation). "
        "A colorful diamond-shaped kite is FIRMLY STUCK and ENTANGLED in the "
        "upper branches of a tall tree in a park. The kite is NOT flying — "
        "it is trapped, hooked on tree limbs, motionless. A thin white kite "
        "string dangles down vertically from the trapped kite. Daylight, "
        "blue sky visible behind tree. Wide shot. Photorealistic, sharp focus. "
        "No people, no animals."
    ),
    "blocked_door_handle": (
        "A realistic photograph (not illustration, not cartoon). Indoor "
        "hallway scene. A wooden bookshelf has fallen sideways and now leans "
        "against a closed wooden interior door, blocking most of it but "
        "leaving a NARROW VERTICAL GAP about 10 cm wide between the bookshelf "
        "edge and the door frame, through which a brass door handle is "
        "clearly visible. A few books on the floor near the bookshelf "
        "(NOT scattered everywhere). Calm aftermath scene, NOT dramatic. "
        "Indoor lighting. Photorealistic. No people."
    ),
    "glass_shard_extract": (
        "A real photograph (not illustration, not cartoon). White kitchen "
        "tile floor seen from top-down view. A clear drinking glass has "
        "shattered, and transparent sharp glass shards are scattered in a "
        "small area. Among the shards lies a small round BLACK hearing aid "
        "button battery (the size of a coin). Sharp focus, bright kitchen "
        "lighting. Photorealistic. NO red color anywhere, NO blood, "
        "NO injury, NO hands, no people."
    ),
}


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    images_dir = project_root / "cases" / "images"
    backup_dir = images_dir / "_backup"
    backup_dir.mkdir(exist_ok=True)

    targets = sys.argv[1:] or list(PROMPTS.keys())
    client = OpenAI(api_key=api_key)

    for name in targets:
        if name not in PROMPTS:
            print(f"  ⚠ unknown task: {name} (available: {list(PROMPTS.keys())})")
            continue

        out_path = images_dir / f"{name}.png"

        if out_path.exists():
            backup_path = backup_dir / f"{name}.png"
            shutil.copy2(out_path, backup_path)
            print(f"  → backed up old image to {backup_path}")

        print(f"\n[generating] {name}")
        print(f"  prompt: {PROMPTS[name][:120]}...")

        response = client.images.generate(
            model="dall-e-3",
            prompt=PROMPTS[name],
            size="1024x1024",
            quality="hd",  # standard → hd: 사실성/디테일 ↑ ($0.04 → $0.08)
            response_format="b64_json",
            n=1,
        )

        b64_png = response.data[0].b64_json
        out_path.write_bytes(base64.b64decode(b64_png))
        print(f"  ✅ saved: {out_path}")

    print("\n완료. 새 이미지 확인 후 5 task 풀 회귀 진행하세요.")


if __name__ == "__main__":
    main()
