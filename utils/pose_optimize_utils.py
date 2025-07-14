import os
import shutil

from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from utils.graphics_utils import getProjectionMatrix
from gaussian_renderer import render
from arguments import ModelParams, get_combined_args, PipelineParams

from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from utils.graphics_utils import fov2focal
from utils.graphics_utils import getWorld2View, getProjectionMatrix

import torch
import torch.nn as nn
import torch.optim as optim
import roma

from tqdm import tqdm
import copy
import argparse

import random
import math
from PIL import Image
import utils.general_utils 
import torchvision.utils
import numpy as np

import sys
import sqlite3
import os
import argparse

class OptimPoseModel(nn.Module):

    def __init__(self, cams:list):
        
        """
        
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        super().__init__()

        print("Optim Pose Model...")
        
        rots = []
        transes = []
        
        self.cams = cams
        for cam in cams:
            cam:Camera    
            rots.append(torch.from_numpy(cam.R).cuda())
            transes.append(torch.from_numpy(cam.T).cuda())

        rots = torch.stack(rots)
        transes = torch.stack(transes)

        rots_unitquat = roma.rotmat_to_unitquat(rots)
        
        self._rot = nn.Parameter(rots_unitquat.float().clone().contiguous().requires_grad_(True)) # [n,4]
        self._trans = nn.Parameter(transes.float().clone().contiguous().requires_grad_(True)) # [n,3]

        
    def forward(self,idx):
        cam: Camera
        cam = copy.deepcopy(self.cams[idx])
        
        quat = self._rot[idx] + 1e-8# [4]
        
        unitquat = quat / quat.norm() # [4]
        rotmat = roma.unitquat_to_rotmat(unitquat[None,:]).squeeze() # [3,3]
        trans = self._trans[idx] # [3]

        cam.world_view_transform = torch.eye(4).cuda()
        cam.world_view_transform[:3,:3] = rotmat.T
        cam.world_view_transform[:3, 3] = trans
        cam.world_view_transform = cam.world_view_transform.transpose(0,1)

        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]

        return cam

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    return cameras_tuples, images_tuples

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

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

def optimize_test_pose(scene: Scene, gaussians:GaussianModel, pipeline:PipelineParams, bg_color:torch.Tensor, kernel_size:float, num_iter_per_view:int=200):
    """
    Run iNeRF-like pose optimization for test veiws.
    Note that test camera pose is not accurate for curve-optimized 3DGS scene, so this process is essential. 
    
    RETURNS
    -------
    optimized_cams: list of Camera object, fit to current scene.
    """
    torch.cuda.empty_cache()

    test_cameras = scene.getTestCameras()
    n = len(test_cameras)

    optim_model = OptimPoseModel(test_cameras)
    optim_param_group = [{"params":[optim_model._rot],   'lr': 5e-5, 'name':"rot"},
                         {"params":[optim_model._trans], 'lr': 5e-4, 'name':"trans"}]

    optimizer = optim.Adam(optim_param_group, lr=5e-4, eps=1e-15)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=num_iter_per_view//20, gamma=0.9)
    
    pbar = tqdm(range(num_iter_per_view), desc="Optimizing...")

    l2_error_ema = 0.0
    
    psnr_best = 0.0
    for iteration in pbar:
        
        idx_list = list(range(n))
        random.shuffle(idx_list)
        
        # Run 1 Epoch.
        psnrs = []
        while len(idx_list) > 0:        
            # Choose one test view.
            idx = idx_list.pop()
            viewpoint_cam = optim_model(idx)
            
            # Loss.
            gt_image = viewpoint_cam.original_image
            image = render(viewpoint_cam, gaussians, pipeline, bg_color, kernel_size)["render"].clamp(0.0, 1.0)
            loss = l1_loss(image, gt_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                l2_error_ema = l2_error_ema * 0.6 + ((gt_image-image)**2).mean().item() * 0.4
            psnr = 20 * math.log10(1.0 / math.sqrt((gt_image-image).pow(2).mean().item()))
            psnrs.append(psnr)
    
        lr_scheduler.step()
        if iteration % 10 == 0:
            with torch.no_grad():
                current_psnr = sum(psnrs) / len(psnrs)
                if current_psnr > psnr_best:
                    psnr_best = current_psnr
                    output = [optim_model(i) for i in range(n)]
                pbar.set_description(f"Optimizing...PSNR ={current_psnr:6.2f} lr = {optimizer.param_groups[0]['lr']:10.6f}")
    return output

@torch.no_grad()
def initialize_test_pose(args:ModelParams, scene: Scene, gaussians:GaussianModel, bg_color:torch.Tensor, exclude = [], old_version=False):
    """
    Only functions when testing without known pose.
    (i.e.) not llffhold-style.
    """
    source_path = args.source_path
    model_path = args.model_path
    if len(scene.getTestCameras()) > 0:
        return
    
    print("Not LLFFHOLD style dataset... Looking for test image without poses.")
    test_image_dir = os.path.join(source_path, "test_images")
    if not os.path.exists(test_image_dir):
        print("No test image detected... Exiting")
        exit()

    # Prepare temporary colmap workspace.
    tmp_colmap_workspace = os.path.join(model_path, "render_colmap")
    shutil.rmtree(tmp_colmap_workspace, ignore_errors=True)
    os.makedirs(tmp_colmap_workspace)

    db_path = os.path.join(tmp_colmap_workspace, "database.db")

    tmp_images = os.path.join(tmp_colmap_workspace, "images_rendered")
    os.makedirs(tmp_images)

    tmp_sparse = os.path.join(tmp_colmap_workspace, "sparse", "1")
    os.makedirs(tmp_sparse)

    flag_EAS = 1

    # Load cams.
    scene.camera_motion_module.load(os.path.join(model_path, "cm.pth"))
    cams = [cam for i,cam in enumerate(scene.camera_motion_module.get_middle_cams()) if i not in exclude]
    
    # Render from train view, save.
    print("Rendering from training view...")
    for i, cam in enumerate(cams):
        cam: MiniCam
        
        # Render and Save.
        render_filename = f"{i:03d}_render.png"
        rendered = render(cam, gaussians, bg_color)["render"]
        # rendered = scene.tone_mapping(rendered)
        torchvision.utils.save_image(rendered, os.path.join(tmp_images, render_filename))

    # Save extrinsic => we will do later to keep track with database order.
    # Save intrinsic in COLMAP convention.
    with open(os.path.join(tmp_sparse, "cameras.txt"), "w") as fp:
        print("# \n"*3, end='', file=fp)
        cam = cams[0]
        w = cam.image_width
        h = cam.image_height
        fx = fov2focal(cam.FoVx, w)
        fy = fov2focal(cam.FoVy, h)
        cx = w/2
        cy = h/2

        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        print(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}", file=fp)        

    # Create Empty pointcloud file.
    with open(os.path.join(tmp_sparse, "points3D.txt"), "w") as fp:
        pass

    # Run colmap with rendered images only.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    
    tmp_sparse_pcd = os.path.join(tmp_colmap_workspace,"sparse","2")
    os.makedirs(tmp_sparse_pcd, exist_ok=True)

    # Save Extrinsic.
    with open(os.path.join(tmp_sparse, "images.txt"), "w") as fp:
        print("# \n"*4, end='', file=fp)
        extr_dic = {}
        for i, cam in enumerate(cams):
            cam: MiniCam
            
            # Render and Save.
            render_filename = f"{i:03d}_render.png"

            # Save pose in COLMAP convention.
            c2w = get_c2w(cam)
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3,:3])
            tvec = w2c[:3,3]
            
            extr_dic[render_filename] = (qvec,tvec)
        _, image_tuples = read_db(db_path=db_path)

        # Follow Database order.
        for i, image_tuple in enumerate(image_tuples):
            render_filename = image_tuple[1]
            qvec, tvec = extr_dic[render_filename]
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            print(i+1, *qvec, *tvec, 1, render_filename, end="\n\n", file=fp)

    # Triangulation. (get PCD)
    inputpath_arg = "--input_path" if not old_version else "--import_path"
    outputpath_arg = "--output_path" if not old_version else "--export_path"
    cmd = f"colmap point_triangulator " + \
              f"--database_path {db_path} " + \
              f"--image_path {tmp_images} " + \
              f"{inputpath_arg} {tmp_sparse} " + \
              f"{outputpath_arg} {tmp_sparse_pcd}"
    do_system(cmd)
    # Prepare test images
    test_image_files = os.listdir(test_image_dir)
    test_image_files.sort()

    tmp_test_images = os.path.join(tmp_colmap_workspace, "test_images")
    shutil.rmtree(tmp_test_images, ignore_errors=True)
    os.makedirs(tmp_test_images)

    for i, test_image_file in enumerate(test_image_files):
        test_image_path = os.path.join(test_image_dir, test_image_file)
        img_pil = Image.open(test_image_path)
        img_pil.save(os.path.join(tmp_test_images,f"{i:03d}.png"))
        print("[DONE]", test_image_path)
    
    # feature extraction and match.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_test_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    

    tmp_sparse_final = os.path.join(tmp_colmap_workspace,"sparse","0")
    shutil.rmtree(tmp_sparse_final, ignore_errors=True)
    os.makedirs(tmp_sparse_final)

    do_system(f"colmap image_registrator "
              f"--database_path {db_path} "
              f"{inputpath_arg} {tmp_sparse_pcd} "
              f"{outputpath_arg} {tmp_sparse_final}")

    tmp_sparse_txt = os.path.join(tmp_colmap_workspace,"sparse_txt")
    shutil.rmtree(tmp_sparse_txt, ignore_errors=True)
    os.makedirs(tmp_sparse_txt)

    do_system(f"colmap model_converter "
              f"--input_path {tmp_sparse_final} "
              f"--output_path {tmp_sparse_txt} "
              f"--output_type TXT")

    do_system(f"python scripts/colmap_visualization.py --path {tmp_colmap_workspace} ")
        
    # Get test images and poses.
    image_txtfile = os.path.join(tmp_sparse_txt, "images.txt")
    
    with open(image_txtfile, 'r') as f:
        lines = f.readlines()

    lines = lines[4:]
    lines = lines[::2]

    test_cams = []

    one_cam = scene.getTrainCameras()[0]
    
    for line in lines:
        tokens = line.strip().split()
        img_name = tokens[-1]
        if "render" in img_name:
            continue

        test_image_path = os.path.join(tmp_test_images, img_name)
        img_pil = Image.open(test_image_path)
        img = utils.general_utils.PILtoTorch(img_pil, img_pil.size).cuda()

        qvec = np.array(list(map(float, tokens[1:5])))
        tvec = np.array(list(map(float, tokens[5:8])))

        R = qvec2rotmat(qvec).T
        T = np.array(tvec)
        
        test_cam = Camera(1, R, T, one_cam.FoVx, one_cam.FoVy, img, None, img_name, 1)
        test_cams.append(test_cam)
    
    scene.test_cameras[1.0] = test_cams