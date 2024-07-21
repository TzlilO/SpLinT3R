import torch
from PIL import Image
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def generate_point_cloud(images, features, intrinsics, extrinsics, threshold=1.0):
    """
    Generate a 3D point cloud from a list of images, their corresponding features, and camera parameters.

    Args:
    images (list of torch.Tensor): List of RGB images.
    features (list of tuple): List of tuples containing keypoints and descriptors for each image.
    intrinsics (list of torch.Tensor): List of camera intrinsic matrices.
    extrinsics (list of torch.Tensor): List of camera extrinsic matrices.
    threshold (float): Threshold for outlier removal based on reprojection error.

    Returns:
    torch.Tensor: 3D point cloud with RGB colors.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize lists to store points and colors
    points_3d = []
    colors = []

    # Extract keypoints and descriptors
    keypoints = [torch.tensor(kp, device=device) for kp, _ in features]
    descriptors = [torch.tensor(desc, device=device) for _, desc in features]

    # Assume images and keypoints are ordered correctly
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            # Match features between image i and j
            matches = match_features(descriptors[i], descriptors[j])

            # Get matched keypoints
            kp1 = keypoints[i][matches[:, 0]]
            kp2 = keypoints[j][matches[:, 1]]

            # Convert keypoints to homogeneous coordinates
            kp1_h = torch.cat([kp1, torch.ones(kp1.shape[0], 1, device=device)], dim=1)
            kp2_h = torch.cat([kp2, torch.ones(kp2.shape[0], 1, device=device)], dim=1)

            # Camera matrices
            P1 = torch.matmul(intrinsics[i], extrinsics[i])
            P2 = torch.matmul(intrinsics[j], extrinsics[j])

            # Triangulate points
            points_4d = triangulate_points(kp1_h, kp2_h, P1, P2)

            # Convert from homogeneous coordinates
            points_3d_curr = points_4d[:, :3] / points_4d[:, 3:]

            # Append points and colors
            points_3d.append(points_3d_curr)
            colors.append(images[i][kp1[:, 1].long(), kp1[:, 0].long()])

    # Concatenate all points and colors
    points_3d = torch.cat(points_3d, dim=0)
    colors = torch.cat(colors, dim=0)

    # Remove outliers based on reprojection error
    points_3d, colors = remove_outliers(points_3d, colors, intrinsics, extrinsics, threshold)

    return torch.cat([points_3d, colors], dim=1)

def match_features(desc1, desc2):
    # Dummy function to match features between two sets of descriptors
    # Replace with your actual feature matching logic
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1.cpu().numpy(), desc2.cpu().numpy(), k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m.queryIdx, m.trainIdx])
    return torch.tensor(good_matches, dtype=torch.long)

def triangulate_points(kp1, kp2, P1, P2):
    # Triangulate points using the Direct Linear Transformation (DLT) method
    num_points = kp1.shape[0]
    A = torch.zeros((num_points, 4, 4), device=kp1.device)

    A[:, 0] = kp1[:, 0].unsqueeze(1) * P1[2] - P1[0]
    A[:, 1] = kp1[:, 1].unsqueeze(1) * P1[2] - P1[1]
    A[:, 2] = kp2[:, 0].unsqueeze(1) * P2[2] - P2[0]
    A[:, 3] = kp2[:, 1].unsqueeze(1) * P2[2] - P2[1]

    _, _, V = torch.svd(A)
    points_4d = V[:, -1]
    return points_4d

def remove_outliers(points_3d, colors, intrinsics, extrinsics, threshold):
    # Reprojection and outlier removal logic
    num_points = points_3d.shape[0]
    reprojection_errors = []

    for i in range(len(intrinsics)):
        P = torch.matmul(intrinsics[i], extrinsics[i])
        points_4d = torch.cat([points_3d, torch.ones(num_points, 1, device=points_3d.device)], dim=1)
        projected_points = torch.matmul(P, points_4d.t()).t()
        projected_points /= projected_points[:, 2].unsqueeze(1)
        reprojection_errors.append(torch.norm(projected_points[:, :2] - keypoints[i], dim=1))

    reprojection_errors = torch.stack(reprojection_errors, dim=1).mean(dim=1)
    mask = reprojection_errors < threshold
    return points_3d[mask], colors[mask]


def project_points_to_image_plane(point_cloud, images, intrinsics, extrinsics):
    """
    Project 3D points onto each camera's image-plane.

    Args:
    point_cloud (torch.Tensor): Tensor of shape (N, 6) where N is the number of points, containing (x, y, z, r, g, b).
    images (list of torch.Tensor): List of RGB images.
    intrinsics (list of torch.Tensor): List of camera intrinsic matrices.
    extrinsics (list of torch.Tensor): List of camera extrinsic matrices.

    Returns:
    list of torch.Tensor: List of projected points for each image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract 3D points from the point cloud
    points_3d = point_cloud[:, :3].to(device)

    # Convert to homogeneous coordinates
    points_3d_h = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1, device=device)], dim=1)

    projected_points_list = []

    for intrinsic, extrinsic in zip(intrinsics, extrinsics):
        # Compute the camera projection matrix
        P = torch.matmul(intrinsic.to(device), extrinsic.to(device))

        # Project 3D points to 2D
        points_2d_h = torch.matmul(P, points_3d_h.t()).t()

        # Convert from homogeneous coordinates to 2D
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]

        projected_points_list.append(points_2d.cpu())

    return projected_points_list


def canny_edge_detection(batch_images):
    """
    Apply Canny edge detection to a batch of RGB PIL images.

    Args:
    batch_images (list of PIL.Image.Image): List of RGB images.

    Returns:
    list of Torch.Tensor: List of images with Canny edge detection applied.
    """
    edge_images = []
    for image in batch_images:
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Apply Canny edge detection
        edges = torch.from_numpy(cv2.Canny(gray, 100, 200))
        # Convert edges to PIL image and append to the list
        edge_images.append(edges)
    return edge_images



def extract_features_sift(batch_images):
    """
    Extract features from a sequence of RGB PIL images using SIFT.

    Args:
    batch_images (list of PIL.Image.Image): List of RGB images of the same scene from different views.

    Returns:
    list of keypoints and descriptors: List containing tuples of keypoints and descriptors for each image.
    """
    sift = cv2.SIFT_create()
    features = []
    for image in batch_images:
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features.append((keypoints, descriptors))
    return features
