
import json
dir_camera = '/home/kiddhcx/Downloads/annotations/Human36M_subject1_camera.json'

with open(dir_camera,'r') as f:
    cameras = json.load(f)

print(cameras)