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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import get_video_cams, ssim_nerf, to8b
from utils.pose_optimize_utils import initialize_test_pose, optimize_test_pose
import imageio
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import lpips as lpips

import warnings
warnings.filterwarnings("ignore")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    error_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"error_map_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(error_map_path, exist_ok=True)

    ssims, psnrs, lpips_scores = [], [], []
    save_images = {"images": [], "depths": []}

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        rendering = torch.clamp(rendering[None, ...], 0., 1.)
        if name == "test":
            save_images["images"].append(rendering.detach().cpu().numpy())
            save_images["depths"].append(depth.detach().cpu().numpy())
        depth = depth - depth.min()
        depth = depth / depth.max()

        gt = view.original_image[0:3, :, :].cuda()
        error_map = (rendering[0] - gt).pow(2).mean(0)
        error_map = error_map / error_map.max()

        cv2.imwrite(os.path.join(error_map_path, '{0:05d}'.format(idx) + ".png"), (error_map.cpu().numpy() * 255).astype(np.uint8))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        psnrs.append(psnr(rendering, gt).item())
        ssims.append(ssim_nerf(rendering, gt))
        lpips_scores.append(lpips_fn(rendering * 2 - 1, gt * 2 - 1).cpu().item())

    if name == 'test':
        print()
        print("MODEL PATH : ", model_path)
        print("PSNR : ", torch.tensor(psnrs).mean().item())
        print("SSIM : ", torch.tensor(ssims).mean().item())
        print("LPIPS: ", torch.tensor(lpips_scores).mean().item())


def render_video(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"video_{scale_factor}")

    makedirs(render_path, exist_ok=True)

    views_video = get_video_cams(views)

    with torch.no_grad():
        render_pkgs = []
        for idx, view in enumerate(tqdm(views_video, desc="Rendering video progress")):
            render_pkgs.append(render(view, gaussians, pipeline, background, kernel_size=kernel_size))

        rgbs = [to8b(render_pkg["render"].permute(1, 2, 0).cpu().numpy()) for render_pkg in render_pkgs]

        depths = [render_pkg["depth"][0].cpu().numpy() for render_pkg in render_pkgs]

        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        depths = depths - depths.min()
        depths = depths / depths.max()
        depths = to8b(depths)

    imageio.mimwrite(os.path.join(render_path, "rgb_video.mp4"), rgbs, fps=60, quality=8)
    imageio.mimwrite(os.path.join(render_path, "depth_video.mp4"), depths, fps=60, quality=8)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video : bool, pose_optimize : bool):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    scale_factor = dataset.resolution
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    with torch.no_grad():
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)
        if not skip_video:
            render_video(dataset.model_path, "video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

    if not skip_test:
        with torch.no_grad():
            print("Rendering Test Set Before Pose Optimization")
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if pose_optimize: # We use the pose optimization code of Deblur-GS
            initialize_test_pose(dataset, scene, gaussians, background, exclude=[], old_version=False)
            fit_cams = optimize_test_pose(scene, gaussians, pipeline=pipeline, bg_color=background, kernel_size=kernel_size)
            with torch.no_grad():
                print("\nRendering Test Set After Pose Optimization")
                render_set(dataset.model_path, "test", scene.loaded_iter, fit_cams, gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--pose_optimize", default=True, type=bool)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.pose_optimize)
