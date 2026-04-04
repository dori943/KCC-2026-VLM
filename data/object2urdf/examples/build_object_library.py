import os
import sys
from object2urdf import ObjectUrdfBuilder

# Build entire libraries of URDFs
# 수정 제안: 절대 경로 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
object_folder = os.path.join(current_dir, "ycb")
builder = ObjectUrdfBuilder(object_folder)
builder.build_library(decompose_concave=False)