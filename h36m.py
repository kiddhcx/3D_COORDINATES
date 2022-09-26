import json
from pycocotools.coco import COCO
import os
import numpy as np
from pose_utils import world2cam, cam2pixel, pixel2cam, rigid_align, process_bbox



dir_joint = '/home/kiddhcx/Downloads/annotations/Human36M_subject1_joint_3d.json'
dir_data = '/home/kiddhcx/Downloads/annotations/Human36M_subject1_data.json'
dir_camera = '/home/kiddhcx/Downloads/annotations/Human36M_subject1_camera.json'

joint_num = 18
joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
lshoulder_idx = joints_name.index('L_Shoulder')
rshoulder_idx = joints_name.index('R_Shoulder')
root_idx = joints_name.index('Pelvis')

def add_thorax(joint_coord):
    thorax = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    thorax = thorax.reshape((1, 3))
    joint_coord = np.concatenate((joint_coord, thorax), axis=0)
    return joint_coord

def load_data(dir_joint, dir_data, dir_camera):
    # aggregate annotations from each subject
    db = COCO()
    cameras = {}
    joints = {}
    
    # data load
    with open(dir_data,'r') as f:
        annot = json.load(f)
    if len(db.dataset) == 0:
        for k,v in annot.items():
            db.dataset[k] = v
    else:
        for k,v in annot.items():
            db.dataset[k] += v
    # camera load
    with open(dir_camera,'r') as f:
        cameras = json.load(f)
    # joint coordinate load
    with open(dir_joint, 'r') as f:
        joints = json.load(f)
    db.createIndex()

    data = []
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        img_width, img_height = img['width'], img['height']

        # camera parameter
        cam_idx = img['cam_idx']
        cam_param = cameras[str(cam_idx)]
        R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            
        # project world coordinate to cam, image coordinate space
        action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
        joint_world = np.array(joints[str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
        joint_world = add_thorax(joint_world)
        joint_cam = world2cam(joint_world, R, t)
        joint_img = cam2pixel(joint_cam, f, c)
        joint_img[:,2] = joint_img[:,2] - joint_cam[root_idx,2]
        joint_vis = np.ones((joint_num,1))
        

        bbox = process_bbox(np.array(ann['bbox']), img_width, img_height)
        if bbox is None: continue
        root_cam = joint_cam[root_idx]
            
        data.append({
            'img_id': image_id,
            'bbox': bbox,
            'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
            'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
            'joint_vis': joint_vis,
            'root_cam': root_cam, # [X, Y, Z] in camera coordinate
            'f': f,
            'c': c})
    
    return data

d = load_data(dir_joint, dir_data, dir_camera)
print(d)