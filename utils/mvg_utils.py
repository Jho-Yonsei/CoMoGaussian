

import numpy as np
from scipy.spatial.transform import Rotation as R

def build_K(fx, fy, cx, cy):
    K = np.array([[fx, 0., cx,],
                  [0., fy, cy,],
                  [0., 0., 1.,]]).astype(np.float64)
    return K

def get_normalized_coords(pixel_coords, K):
    
    pixel_coords_homog = np.pad(pixel_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)    
    
    K_inv = np.linalg.inv(K)
    norm_coords = K_inv @ pixel_coords_homog.T

    return (norm_coords.T)[...,:2] 


def normalized_coords_to_cam_coords(normalized_coords, depth):
    normalized_coords_homog = np.pad(normalized_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw, 3)
    return normalized_coords_homog * depth.reshape(-1,1)

def cam_to_world_coords(cam_coords, c2w):
    """
    cam_coords: (hw, 3)
    c2w: (4,4)
    """
    cam_coords_homog = np.pad(cam_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw,4)
    world_coords = c2w @ cam_coords_homog.T

    return (world_coords.T)[...,:3]



def to_w2c(c2w):
    """
    build w2c 4x4 matrix from c2w ((3x4) or (4x4))
    """

    w2c = np.eye(4,4)
    
    cam_loc = c2w[:3,3]
    rot_c2w = c2w[:3,:3]

    rot_w2c = rot_c2w.T
    trans_vec = -rot_w2c@cam_loc

    w2c[:3, :3] = rot_w2c
    w2c[:3,  3] = trans_vec
    w2c = w2c.astype(np.float64)
    return w2c

def mean_camera_pose(c2ws):
    """
    Compute the mean camera pose from a list of SE(3) matrices.

    Parameters:
    - se3_matrices (numpy.ndarray): Array of SE(3) matrices of shape (n, 4, 4), where n is the number of matrices.

    Returns:
    - numpy.ndarray: Mean SE(3) matrix representing the averaged camera pose.
    """
    translations = c2ws[:, :3, 3]  # Extract translation vectors
    rotations = R.from_matrix(c2ws[:, :3, :3])  # Extract rotation matrices

    # Compute mean translation
    mean_translation = np.mean(translations, axis=0)

    # Compute mean rotation
    mean_rotation = rotations.mean().as_matrix()

    # Construct mean SE(3) matrix
    mean_se3_matrix = np.eye(4)
    mean_se3_matrix[:3, :3] = mean_rotation
    mean_se3_matrix[:3, 3] = mean_translation

    return mean_se3_matrix


def get_c2w_from_eye(eye, lookat, up):
    # get c2w matrix for pivot camera.
    z_vec = lookat-eye
    x_vec = np.cross(up,z_vec) 
    y_vec = np.cross(z_vec,x_vec)

    x_vec = x_vec/np.linalg.norm(x_vec)
    y_vec = y_vec/np.linalg.norm(y_vec)
    z_vec = z_vec/np.linalg.norm(z_vec)
    
    rot_pivot = np.stack([x_vec,y_vec,z_vec],axis=0).T
    
    c2w = np.eye(4)
    c2w[:3,:3] = rot_pivot
    c2w[:3,3] = eye
    return c2w