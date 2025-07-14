from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from scene.cameras import Camera
from skimage import metrics

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def ssim_nerf(img0, img1):
    img0 = (img0 * 2 - 1).clamp(-1, 1)
    img1 = (img1 * 2 - 1).clamp(-1, 1)
    img0 = img0.squeeze().permute(1, 2, 0).cpu().numpy()
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    value, ssimmap = metrics.structural_similarity(img0, img1, multichannel=True, channel_axis=-1, full=True)
    return value

def get_video_cams(views):
    cam = views[0]
    views_video = []

    for idx in range(len(views) - 1):
        Rs, Ts = interpolate_SE(views[idx].R, views[idx].T, views[idx + 1].R, views[idx + 1].T)
        for k in range(len(Rs)):
            view = Camera(colmap_id=cam.colmap_id, R=Rs[k], T=Ts[k], 
                            FoVx=cam.FoVx, FoVy=cam.FoVy, 
                            image=cam.image, gt_alpha_mask=cam.gt_alpha_mask,
                            image_name=cam.image_name, uid=cam.uid, data_device=cam.data_device)
            views_video.append(view)

    return views_video

def interpolate_SE(R1, T1, R2, T2, num_points=40):

    slerp = Slerp([0, 1], Rotation.from_matrix([R1, R2]))

    interp_points = np.linspace(0, 1, num_points)

    Rs, Ts = [], []
    for alpha in interp_points:
        t = (1 - alpha) * T1 + alpha * T2
        R_interp = slerp(alpha).as_matrix()
        Rs.append(R_interp)
        Ts.append(t)
    return Rs, Ts