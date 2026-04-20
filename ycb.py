from pathlib import Path

YCB_DIR = "/workspace/KCC-2026-VLM/data/object2urdf/examples/ycb"

# 1. Path 객체 생성
base_path = Path(YCB_DIR)

# 2. rglob(recursive glob)을 사용하여 모든 하위 폴더의 .urdf 탐색
urdf_files = list(base_path.rglob("*.urdf"))

if not urdf_files:
    print(f"❌ {YCB_DIR} 하위에 URDF 파일이 없습니다.")
else:
    print(f"✅ 총 {len(urdf_files)}개의 URDF 파일을 찾았습니다:\n")
    for urdf in urdf_files:
        # 파일명만 출력하고 싶다면 urdf.name, 전체 경로는 urdf
        print(urdf)