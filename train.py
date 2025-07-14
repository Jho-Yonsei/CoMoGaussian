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

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.como_kernel import CoMoKernel
from utils.general_utils import safe_state
from utils.visualization import Visualizer
from utils.render_utils import get_video_cams, ssim_nerf, to8b
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import imageio
from arguments import ModelParams, PipelineParams, OptimizationParams, CoMoParams
import lpips.lpips as lpips
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
import warnings
warnings.filterwarnings("ignore")

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, comoopt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras

    como_kernel = CoMoKernel(num_views=len(trainCameras),
                             view_dim=comoopt.view_dim,
                             num_warp=comoopt.num_warp,
                             method=comoopt.method,
                             adjoint=comoopt.adjoint,
                             iteration=opt.iterations).cuda()
    
    visualizer = Visualizer(opt, scene, gaussians, background, trainCameras, dataset, pipe, como_kernel, vis_cam_idx=None)
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy() 

        camera_index = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(camera_index)
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset = None

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset, compute_grad_cov2d=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        pixel_mask_loss = 0.
        ortho_loss = 0.
        
        if iteration > comoopt.start_warp:
            
            image_ori = image
            warped_cams, ortho_loss = como_kernel.get_warped_cams(viewpoint_cam)

            rendered_images = []
            for i, cam in enumerate(warped_cams):
                render_pkg = render(cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset, compute_grad_cov2d=True)
                rendered_images.append(render_pkg["render"])

            if iteration > comoopt.start_pixel_weight:
                rendered_images = torch.stack(rendered_images, dim=0)
                pixel_weights, pixel_mask = como_kernel.get_weight_and_mask(rendered_images.detach(), viewpoint_cam.uid)
                image = torch.sum(pixel_weights * rendered_images, dim=0)
                image = image * pixel_mask + image_ori * (1 - pixel_mask)
                pixel_mask_loss = pixel_mask.mean()

            else:
                rendered_images.append(image_ori)
                rendered_images = torch.stack(rendered_images, dim=0)
                image = torch.mean(rendered_images, dim=0)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + pixel_mask_loss * 1e-3 + ortho_loss * 1e-4 
        loss.backward()


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size), dataset.model_path)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)


            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration > comoopt.start_warp:
                    como_kernel.optimizer.step()
                    como_kernel.optimizer.zero_grad()
                    como_kernel.adjust_lr()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 1000 == 0 and iteration > 1000:
                visualizer.run(iteration)

    with torch.no_grad():
        viewpoint_stack = scene.getTrainCameras().copy()
        scene_name = os.path.join(scene.model_path, "images")
        os.makedirs(scene_name, exist_ok=True)

        for i, cam in enumerate(viewpoint_stack):
            render_pkg = render(cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, compute_grad_cov2d=False)
            image = render_pkg["render"]
            image_ori = image
            cv2.imwrite("./{}/image_{}.png".format(scene_name, i), (image[[2, 1, 0]].clip(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            warped_cams, _ = como_kernel.get_warped_cams(cam)

            rendered_images = []
            for j, cam_w in enumerate(warped_cams):
                image_warp = render(cam_w, gaussians, pipe, background, kernel_size=dataset.kernel_size, compute_grad_cov2d=False)["render"]
                rendered_images.append(image_warp)
                cv2.imwrite("./{}/image_{}_warp_{}.png".format(scene_name, i, j), (image_warp[[2, 1, 0]].clip(0, 1).detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            rendered_images = torch.stack(rendered_images, dim=0)
            pixel_weights, pixel_mask = como_kernel.get_weight_and_mask(rendered_images.detach(), cam.uid)
            image = torch.sum(pixel_weights * rendered_images, dim=0)
            image = image * pixel_mask + image_ori * (1 - pixel_mask)
            cv2.imwrite("./{}/image_{}_blur.png".format(scene_name, i), (image[[2, 1, 0]].clip(0, 1).detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            cv2.imwrite("./{}/image_{}_mask.png".format(scene_name, i), (pixel_mask.clip(0, 1).detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            cv2.imwrite("./{}/image_{}_gt.png".format(scene_name, i), (cam.original_image[[2, 1, 0]].clip(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(args.model_path+"/TEST", exist_ok = True)
    os.makedirs(args.model_path+"/TRAIN", exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, savedir):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'Test', 'cameras' : scene.getTestCameras()},)

        for config in validation_configs:
            _type = config["name"].upper()
            if _type == "TEST":
                print("[ITER {}] NUM GAUSSIANS  : {}".format(iteration, scene.gaussians.get_xyz.shape[0]))
                with open(f"{savedir}/log.txt", "a") as f:
                    f.write("[ITER {}] NUM GAUSSIANS  : {} \n".format(iteration, scene.gaussians.get_xyz.shape[0]))
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim_nerf(image, gt_image)
                    lpips_test += lpips_fn(image * 2 - 1, gt_image * 2 - 1).item()

                    imageio.imwrite(f"{savedir}/{_type}/img_{iteration}_{idx:03d}.png", (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    if iteration == testing_iterations[0]:
                        imageio.imwrite(f"{savedir}/{_type}/GT_{idx:03d}.png", (gt_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print("[ITER {}] Evaluating {}: L1   {:.4f} PSNR  {:.2f} ".format(iteration, config['name'], l1_test, psnr_test))
                print("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], ssim_test, lpips_test))
                with open(f"{savedir}/psnr.txt", "a") as f:
                    f.write("[ITER {}] Evaluating {}: L1   {:.4f} PSNR  {:.2f}  \n".format(iteration, config['name'], l1_test, psnr_test))
                    f.write("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}  \n".format(iteration, config['name'], ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    cp = CoMoParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4_000, 8_000, 12_000, 16_000, 20_000, 24_000, 28_000, 32_000, 36_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000, 8_000, 12_000, 16_000, 20_000, 24_000, 28_000, 32_000, 36_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), cp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
