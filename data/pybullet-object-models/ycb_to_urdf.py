import os
import pybullet_object_models
from pybullet_object_models import ycb_objects
from pybullet_object_models import graspa_layouts
from pybullet_object_models import superquadric_objects

# Get the path to the objects inside each package

print(ycb_objects.getDataPath())
print(graspa_layouts.getDataPath())
print(superquadric_objects.getDataPath())

# With this path you can access to the object files, e.g. its URDF model

obj_name = 'YcbBanana' # name of the object folder

path_to_urdf = os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf")