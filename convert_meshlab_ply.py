import numpy as np
from plyfile import PlyData
import os
import open3d as o3d
import ArgumentParser


C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert2meshlab(path, name):

    plydata = PlyData.read(os.path.join(path, name))

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    rgb = SH2RGB(features_dc[..., 0])
    # clamp the lower bound of rgb values to 0
    rgb = np.maximum(rgb, 0)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = sigmoid(opacities)
    opacity_mask = (opacities > 0.005).squeeze(1)
    xyz = xyz[opacity_mask]
    rgb = rgb[opacity_mask]

    # for point with rgb values large than 1, we need to rescale all channels by making the largest channel 1
    max_rgb = np.max(rgb, axis=1)
    max_rgb = np.maximum(max_rgb, 1)
    rgb = rgb / max_rgb[:, np.newaxis]

    # for checking
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(os.path.join(path, "pcd_" + name), pcd)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script parameters")
    parser.add_argument("--path", type=str, default="./")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    convert2meshlab(args.path, args.name)