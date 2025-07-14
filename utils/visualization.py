
import os
from scene.cameras import Camera, get_c2w, c2w_to_cam
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams

# from scene.motion import CameraMotionModule
import torch
import torchvision
import torch.nn as nn
import numpy as np
from gaussian_renderer import render
import math
import cv2
import matplotlib.pyplot as plt
# import render_spiral
import shutil
# import utils.colorize
import open3d as o3d
import utils.mvg_utils

class Visualizer:
    
    def __init__(self, opt:OptimizationParams, scene: Scene, gaussians: GaussianModel, 
                 bg_color, trainCameras, dataset, pipe, como_kernel,
                 vis_cam_idx = None):
        
        self.gaussians = gaussians
        self.ref_camera = self._get_visualization_camera(scene, gaussians, vis_cam_idx)
        self.bg_color = bg_color
        self.draw_camera = True
        self.dataset = dataset
        self.pipe = pipe
        self.como_kernel = como_kernel
        self.trainCameras = trainCameras
        
        self.traj_vis_path = os.path.join(scene.model_path, "vis_traj")
        shutil.rmtree(self.traj_vis_path, ignore_errors=True)
        os.makedirs(self.traj_vis_path, exist_ok=True)

    @torch.no_grad()
    def draw_cone_on_render_img(self, cam_render: Camera, rendered_img:np.ndarray, cams_for_draw:list, traincam: Camera, scale=1.0, color=np.array([0,0,255])):
        """
        Draw camera cone on the rendered_img, which is rendered from cam.
        
        ARGUMENTS
        ---------
        cam_render: Camera object used for render 'rendered_img'
        rendered_img: np.array (H,W,3)
        cam_for_draw: Camera object to be painted. 
        scale: float. Decides how large the cone is.
        color: RGB format
        RETURNS
        -------
        rendered_img with cam cone.
        """
        if not self.draw_camera:
            return rendered_img
        
        color = np.ascontiguousarray(color[::-1]) # to BGR format for cv2
        H,W,_ = rendered_img.shape
        
        c2ws_draw = np.stack([get_c2w(cam) for cam in cams_for_draw])
        for cam_draw, c2w_draw in zip(cams_for_draw, c2ws_draw):
            cone_x, cone_y = math.tan(cam_draw.FoVx/2), math.tan(cam_draw.FoVy/2)

            cone_camera_draw_space_homog = np.pad(np.array([[ 0.0   ,   0.0  , 0.0],
                                                            [ cone_x,  cone_y, 1.0],
                                                            [ cone_x, -cone_y, 1.0], 
                                                            [-cone_x, -cone_y, 1.0],
                                                            [-cone_x,  cone_y, 1.0]]) * scale , 
                                                ((0,0),(0,1)),
                                                'constant',
                                                constant_values=1.0) # (5,4)
            
            cone_world_space_homog = cone_camera_draw_space_homog @ c2w_draw.T # (5,4)
            
            cam_hom = cone_camera_draw_space_homog @ cam_render.world_view_transform.cpu().numpy() # (5,4)
            if (cam_hom[:,2]/cam_hom[:,3] < 0.1).any():
                continue
            
            ndc_hom = cone_world_space_homog @ cam_render.full_proj_transform.cpu().numpy() # (5,4)
            ndc = ndc_hom[:,:3] / ndc_hom[:,3:] # [5,3]

            pix = (( ndc[:,:2] + 1.0) * np.array([W,H]).astype(float) -1.0) * 0.5 # [5,2]
            connectivity = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            for i,j in connectivity:
                try:
                    rendered_img = cv2.line(rendered_img, pix[i].astype(int).tolist(), pix[j].astype(int).tolist(), color.tolist(), thickness=1)
                except Exception as e:
                    pass
                
        return rendered_img

    @torch.no_grad()
    def draw_cone_on_render_img_warp(self, cam_render: Camera, rendered_img:np.ndarray, cams_for_draw:list, traincam: Camera, scale=0.5, color=np.array([255,0,0])):
        """
        Draw camera cone on the rendered_img, which is rendered from cam.
        
        ARGUMENTS
        ---------
        cam_render: Camera object used for render 'rendered_img'
        rendered_img: np.array (H,W,3)
        cam_for_draw: Camera object to be painted. 
        scale: float. Decides how large the cone is.
        color: RGB format
        RETURNS
        -------
        rendered_img with cam cone.
        """
        if not self.draw_camera:
            return rendered_img
        
        color = np.ascontiguousarray(color[::-1]) # to BGR format for cv2
        H,W,_ = rendered_img.shape

        start_color = color
        end_color = np.array([255, 230, 230])
        
        c2ws_draw = np.stack([get_c2w(cam) for cam in cams_for_draw])
        for idx, (cam_draw, c2w_draw) in enumerate(zip(cams_for_draw, c2ws_draw)):
            cone_x, cone_y = math.tan(cam_draw.FoVx/2), math.tan(cam_draw.FoVy/2)

            cone_camera_draw_space_homog = np.pad(np.array([[ 0.0   ,   0.0  , 0.0],
                                                            [ cone_x,  cone_y, 1.0],
                                                            [ cone_x, -cone_y, 1.0], 
                                                            [-cone_x, -cone_y, 1.0],
                                                            [-cone_x,  cone_y, 1.0]]) * scale , 
                                                ((0,0),(0,1)),
                                                'constant',
                                                constant_values=1.0) # (5,4)
            
            cone_world_space_homog = cone_camera_draw_space_homog @ c2w_draw.T # (5,4)
            
            cam_hom = cone_camera_draw_space_homog @ cam_render.world_view_transform.cpu().numpy() # (5,4)
            if (cam_hom[:,2]/cam_hom[:,3] < 0.1).any():
                continue
            
            ndc_hom = cone_world_space_homog @ cam_render.full_proj_transform.cpu().numpy() # (5,4)
            ndc = ndc_hom[:,:3] / ndc_hom[:,3:] # [5,3]

            pix = (( ndc[:,:2] + 1.0) * np.array([W,H]).astype(float) -1.0) * 0.5 # [5,2]
            connectivity = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            t = idx / (len(cams_for_draw) - 1)  # Interpolation ratio (0 to 1)
            color = (1 - t) * start_color + t * end_color  # Linear interpolation
            color = color.astype(int)
            for i,j in connectivity:
                try:
                    rendered_img = cv2.line(rendered_img, pix[i].astype(int).tolist(), pix[j].astype(int).tolist(), color.tolist(), thickness=1)
                except Exception as e:
                    pass
                
        return rendered_img
    


    def _get_visualization_camera(self, scene:Scene, gaussians:GaussianModel, vis_cam_idx=None, threshold = 0.5):
        """
        obtain "reasonable" camera to watch observation process.
        """
        
        if vis_cam_idx is not None:
            self.draw_camera = False
            return self.sample_subframe_cams(idx=vis_cam_idx, num_subframes=1)[0]
        
        print(" ==> searching for reasonable camera")
        
        lookat = gaussians._xyz.detach().cpu().numpy().mean(axis=0)
        pts = np.stack([cam.camera_center.cpu().numpy() for cam in scene.getTrainCameras()]) # (n,3)
        
        # Binary search for the lowest zoom which can see all cameras.

        # zoom_lb, zoom_ub = 1.5, 100.0
        zoom_lb, zoom_ub = 0.01, 3.0

        while zoom_ub - zoom_lb >= 1e-3:
            zoom = (zoom_lb + zoom_ub) / 2.0
            c2ws = np.stack([get_c2w(cam) for cam in scene.getTrainCameras()])
            mean_c2w = utils.mvg_utils.mean_camera_pose(c2ws)
            eye = mean_c2w[:3,3]
            up = mean_c2w[:3,1]
            zoomout_eye = lookat + zoom * (eye-lookat)
            zoomout_c2w = utils.mvg_utils.get_c2w_from_eye(zoomout_eye,lookat,up)
            zoomout_cam = c2w_to_cam(ref_cam=scene.getTrainCameras()[0], c2w=zoomout_c2w)
            W,H  = zoomout_cam.image_width, zoomout_cam.image_height

            pts_hom = np.pad(pts, ((0,0),(0,1)), 'constant', constant_values=1.0) # (n,4)
            pts_cam_hom = pts_hom @ zoomout_cam.world_view_transform.cpu().numpy() # (n,4)
            pts_cam = pts_cam_hom[:,:3] / pts_cam_hom[:,3:] # (n,3)
            pts_cheirality = pts_cam[:,2] >= 0.1 # (n,)

            pts_ndc_hom = pts_hom @ zoomout_cam.full_proj_transform.cpu().numpy() # (n,4)
            pts_ndc = pts_ndc_hom[:,:3] / pts_ndc_hom[:,3:] # (n,3)
            
            pts_pix = (( pts_ndc[:,:2] + 1.0) * np.array([zoomout_cam.image_width,zoomout_cam.image_height]).astype(float) -1.0) * 0.5 # (n,2)

            pts_inside = np.logical_and( np.logical_and( pts_pix[:,0] >= -threshold*W , pts_pix[:,0] <= (1.0+threshold)*W) , 
                                         np.logical_and( pts_pix[:,1] >= -threshold*H , pts_pix[:,1] <= (1.0+threshold)*H) )
            
            pts_good = np.logical_and(pts_inside, pts_cheirality)

            if pts_good.all():
                zoom_ub = zoom
            else:
                zoom_lb = zoom
        
        return zoomout_cam

    @torch.no_grad()
    def render_gaussian_and_cams(self, iteration):
        rendered = render(self.ref_camera, self.gaussians, self.pipe, self.bg_color, kernel_size=self.dataset.kernel_size, subpixel_offset=None, compute_grad_cov2d=True)["render"].permute(1,2,0).cpu().numpy()
        rendered = np.ascontiguousarray((rendered * 255.0).clip(0.0,255.0).astype(np.uint8)[:,:,::-1])
        
        color1 = np.array([0,255,255])
        color2 = np.array([255,255,0])
        t = np.linspace(0, 1, len(self.trainCameras))[:,None]
        colors = ((1 - t) * color1 + t * color2).astype(np.uint8)
        
        for i ,color in enumerate(colors):
            subframe_cams = [self.trainCameras[i]]
            rendered = self.draw_cone_on_render_img(self.ref_camera, rendered, subframe_cams, self.trainCameras[i], scale=0.5, color=color)

        cv2.imwrite(os.path.join(self.traj_vis_path, f"{iteration:05d}.png" ), rendered) # RGB2BGR

    @torch.no_grad()
    def render_gaussian_and_cams_warp(self, iteration):
        rendered = render(self.ref_camera, self.gaussians, self.pipe, self.bg_color, kernel_size=self.dataset.kernel_size, subpixel_offset=None, compute_grad_cov2d=True)["render"].permute(1,2,0).cpu().numpy()
        rendered = np.ascontiguousarray((rendered * 255.0).clip(0.0, 255.0).astype(np.uint8)[:,:,::-1])
        
        color1 = np.array([0, 255, 255])
        color2 = np.array([255, 255, 0])
        t = np.linspace(0, 1, len(self.trainCameras))[:,None]
        colors = ((1 - t) * color1 + t * color2).astype(np.uint8)
        
        for i ,color in enumerate(colors):
                
            rendered = render(self.ref_camera, self.gaussians, self.pipe, self.bg_color, kernel_size=self.dataset.kernel_size, subpixel_offset=None, compute_grad_cov2d=True)["render"].permute(1,2,0).cpu().numpy()
            rendered = np.ascontiguousarray((rendered * 255.0).clip(0.0, 255.0).astype(np.uint8)[:,:,::-1])

            warped_cams = self.como_kernel.get_warped_cams(self.trainCameras[i])[0]
            subframe_cams = warped_cams
            rendered = self.draw_cone_on_render_img_warp(self.ref_camera, rendered, subframe_cams, self.trainCameras[i])

            cv2.imwrite(os.path.join(self.traj_vis_path, f"{iteration:05d}_warp_{self.trainCameras[i].image_name}.png" ), rendered) # RGB2BGR

    def run(self, current_iter):
            self.render_gaussian_and_cams(current_iter)
            self.render_gaussian_and_cams_warp(current_iter)



