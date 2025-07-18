#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View2_torch, getProjectionMatrix
import copy

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image = image
        self.image_name = image_name
        self.gt_alpha_mask = gt_alpha_mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if type(R) == np.ndarray:
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        else:
            self.world_view_transform = getWorld2View2_torch(R, T, torch.tensor(trans), scale).transpose(0, 1)
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = torch.inverse(self.world_view_transform)[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
         
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def get_c2w(cam:Camera, want_numpy=True):
    """
    Get MVG convention c2w matrix from Camera object.
    
    ARGUMENTS
    ---------
    cam: camera object
    want_numpy: If True, returns numpy, otherwise tensor object.

    RETURNS
    -------
    c2w: (4,4) np array or [4,4] tensor, depending on your option.
    """
    if want_numpy:
        c2w = np.eye(4)
        c2w[:3,:3] = cam.world_view_transform[:3,:3].cpu().numpy()
        c2w[:3,3] = cam.camera_center.cpu().numpy()
    else:
        raise NotImplementedError
    
    return c2w

def c2w_to_cam(ref_cam:Camera, c2w):
    
    device = ref_cam.world_view_transform.device
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    
    rot = c2w[:3,:3]
    trans = c2w[:3,3]

    world_view_transform = torch.eye(4, device=device)
    world_view_transform[:3,:3] = rot # NOTE rot.T.T 
    world_view_transform[3,:3] = -trans@rot # NOTE: not [:3,3] for world-view transform.
    
    
    cam = copy.deepcopy(ref_cam)
        
    cam.world_view_transform = world_view_transform
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
   
    return cam
