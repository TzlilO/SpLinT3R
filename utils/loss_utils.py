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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_smoothness_and_consistency_loss1(normals):
    # Compute the normal consistency loss
    # normal_dot_products = torch.einsum('ij,kj->ik', normals, normals)  # Shape (N, N)
    # normal_consistency = (1 - normal_dot_products).mean()
    # normal_dot_products = torch.einsum('bijc,bklc->bijkl', normals, normals)
    normal_dot_products = torch.einsum('bijc,bklc->b', normals, normals).sqrt()
    normal_consistency = normal_dot_products.mean()
    return normal_consistency


def compute_smoothness_and_consistency_loss(normals):
    M, H, W, C = normals.shape

    # Compute normal dot products within each patch, preserving spatial arrangement
    normal_dot_products = torch.einsum('bhwc,bHWc->bhwHW', normals, normals)  # Shape (M, H, W, H, W)

    # Compute distances between points in the grid (euclidean distance)
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).float()  # Shape (H, W, 2)
    coords = coords.view(1, H, W, 1, 1, 2) - coords.view(1, 1, 1, H, W, 2)  # Shape (1, H, W, H, W, 2)
    distances = torch.norm(coords, dim=-1)  # Shape (1, H, W, H, W)

    # Avoid division by zero by adding a small epsilon to the distances
    epsilon = 1e-6
    distances = distances + epsilon

    # Compute the weighted normal consistency loss
    sig = 1
    weights = torch.exp(-distances/sig**2)  # Using a Gaussian kernel for weighting
    normal_consistency = (1 - normal_dot_products) * weights.cuda()  # Shape (M, H, W, H, W)
    normal_consistency_loss = normal_consistency.mean(dim=(1, 2, 3, 4))  # Mean over the grid

    # Return the mean normal consistency loss over all patches
    return normal_consistency_loss.mean()


def compute_cosine_similarity(grad_tensor):
    """
    Computes the cosine similarity between consecutive batches in a tensor of shape MxNx4x4x3
    using efficient tensor operations.

    Parameters:
        grad_tensor (torch.Tensor): A tensor of shape MxNx4x4x3 containing gradients.

    Returns:
        torch.Tensor: A tensor of cosine similarity scores between each consecutive batch pair.
    """
    # Flatten the tensors from Nx4x4x3 to a single vector per batch
    # grad_tensor_flat = grad_tensor.view(grad_tensor.shape[0], -1)

    # Extract the vectors for the first M-1 batches and the last M-1 batches
    batch1 = grad_tensor[:-1]  # From first to second-to-last
    batch2 = grad_tensor[1:]  # From second to last

    # Compute the dot product of corresponding vectors
    dot_product = (batch1 * batch2).sum(dim=0)

    # Compute norms of the batch vectors
    norm1 = batch1.norm(dim=0)
    norm2 = batch2.norm(dim=0)

    # Calculate cosine similarity: cos(theta) = dot(a, b) / (norm(a) * norm(b))
    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity


def geometry_loss(surface_normals, areas, weight=1):
    """
    Computes the cosine similarity between nearby normals on a surface per patch.

    Parameters:
        surface_normals (torch.Tensor): Patches of scene's sureface (shape: BATCH_SIZE x res x res x 3).
        areas (torch.Tensor): Patches areas (shape: BATCH_SIZE x res x res).
    Returns:
        torch.Tensor: Loss value (scalar).
    """
    # Shift normals to get neighboring normals
    normals_shift_u = torch.roll(surface_normals, shifts=-1, dims=1)
    normals_shift_v = torch.roll(surface_normals, shifts=-1, dims=2)

    # Compute cosine similarity between neighboring normals
    cos_sim_u = F.cosine_similarity(surface_normals, normals_shift_u, dim=-1)
    cos_sim_v = F.cosine_similarity(surface_normals, normals_shift_v, dim=-1)

    # Aggregate the loss (1 - cosine similarity)
    loss_u = (1 - cos_sim_u)
    loss_v = (1 - cos_sim_v)

    loss = (loss_u + loss_v) / areas  # patches with larger area will be weighted less
    return loss.mean() * weight


def TV_SH_loss(sh_coeffs):
    """
    Compute the total variation (TV) loss for spherical harmonics coefficients.

    Parameters:
        sh_coeffs (torch.Tensor): Tensor of shape (B, 4, 4, C) representing the SH coefficients of the surface patches.

    Returns:
        torch.Tensor: Total variation loss.
    """
    B, udim, vdim, C = sh_coeffs.shape

    # Compute finite differences along u and v directions
    delta_u_sh = sh_coeffs[:, 1:, :, :] - sh_coeffs[:, :-1, :, :]
    delta_v_sh = sh_coeffs[:, :, 1:, :] - sh_coeffs[:, :, :-1, :]

    # Compute the total variation loss
    tv_loss_u = torch.sum(torch.sqrt(torch.sum(delta_u_sh ** 2, dim=-1) + 1e-6))
    tv_loss_v = torch.sum(torch.sqrt(torch.sum(delta_v_sh ** 2, dim=-1) + 1e-6))

    tv_loss = tv_loss_u + tv_loss_v

    return tv_loss / (B * udim * vdim * C)


def TV_loss(du, dv):
    """
    Compute the total variation (TV) loss for spline surface patches.

    Parameters:
        control_points (torch.Tensor): Tensor of shape (B, 4, 4, 3) representing the control points of the surface patches.

    Returns:
        torch.Tensor: Total variation loss.
    """
    B, udim, vdim, _ = du.shape


    # Compute the total variation loss
    tv_loss = torch.sum(torch.sqrt(du ** 2 + 1e-6)) + torch.sum(torch.sqrt(dv ** 2 + 1e-6))

    return tv_loss / (B * udim * vdim)
