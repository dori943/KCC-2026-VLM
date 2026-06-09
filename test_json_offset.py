"""
test_json_offset.py
assembly_manager의 JSON step 정보 확인
"""

import json
from assembly_manager import AssemblyManager

# module3_balloon_output.json 로드
json_path = "/workspace/KCC-2026-VLM/module3_balloon_output.json"

with open(json_path, 'r') as f:
    data = json.load(f)

print("[Test] JSON assembly_steps:")
for step_data in data.get("assembly_steps", []):
    step_no = step_data.get("step")
    base = step_data.get("base_object")
    attach = step_data.get("attach_object")
    offset = step_data.get("relative_offset_from_base")
    
    print(f"\nStep {step_no}:")
    print(f"  Base: {base}")
    print(f"  Attach: {attach}")
    print(f"  Relative offset: {offset}")
    
    if base == "spatula" and attach == "gelatin_box":
        print(f"  ✓ This is the assembly step we need!")

# AssemblyManager로 로드
print("\n" + "="*60)
print("[Test] AssemblyManager._plan:")

assembly_mgr = AssemblyManager()
assembly_mgr.load_plan_from_json(json_path)

_plan = assembly_mgr._plan
if _plan and hasattr(_plan, 'steps'):
    for step in _plan.steps:
        step_no = step.step
        base = step.base_object
        attach = step.attach_object
        offset = step.attachment_position
        
        print(f"\nStep {step_no}:")
        print(f"  Base: {base}")
        print(f"  Attach: {attach}")
        print(f"  Attachment position: {offset}")
        
        if base == "spatula" and attach == "gelatin_box":
            print(f"  ✓ This is the assembly step we need!")
            print(f"  Expected in URDF joint: xyz=\"{offset[0]} {offset[1]} {offset[2]}\"")
