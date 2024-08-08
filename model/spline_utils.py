import torch
from typing import NamedTuple

import wandb
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.special import comb


def bernstein_poly(i, n, t, device='cuda'):
    return torch.tensor(comb(n, i) * (t ** i) * ((1 - t) ** (n - i)), device=device)


def basis_functions(n, t):
    return torch.cat([bernstein_poly(i, n, t).unsqueeze(0) for i in range(n + 1)], dim=0)


def compute_face_points(xyz):
    return (xyz[:, :-1, :-1] + xyz[:, :-1, 1:] + xyz[:, 1:, :-1] + xyz[:, 1:, 1:]) * 0.25


def compute_edge_points(xyz, face_points):
    N, H, W, C = xyz.shape
    edge_points = torch.zeros((N, 2 * H - 1, 2 * W - 1, C), dtype=xyz.dtype, device=xyz.device)
    edge_points[:, 1::2, ::2] = (xyz[:, :-1, :] + xyz[:, 1:, :]) * 0.5
    edge_points[:, ::2, 1::2] = (xyz[:, :, :-1] + xyz[:, :, 1:]) * 0.5
    edge_points[:, 1::2, 1::2] = face_points
    return edge_points


def compute_vertex_points(xyz, face_points, edge_points):
    N, H, W, C = xyz.shape
    vertex_points = torch.zeros((N, 2 * H - 1, 2 * W - 1, C), dtype=xyz.dtype, device=xyz.device)
    vertex_points[:, ::2, ::2] = xyz
    vertex_points[:, 1::2, ::2] = edge_points[:, 1::2, ::2]
    vertex_points[:, ::2, 1::2] = edge_points[:, ::2, 1::2]
    vertex_points[:, 1::2, 1::2] = face_points
    return vertex_points


def catmull_clark_subdivision(xyz):
    """
    Given a BSpline patch (N patches in total), where each patch is defined by 4x4 3D control points,
    this function returns 4 new patches for each of the N patches using the Catmull-Clark subdivision algorithm.

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing control points in 3D.

    Returns:
        torch.Tensor: Subdivided 4 new patches of the old patch according to its 3D control points tensor of shape (4N, 4, 4, 3).
    """
    face_points = compute_face_points(xyz)
    edge_points = compute_edge_points(xyz, face_points)
    vertex_points = compute_vertex_points(xyz, face_points, edge_points)

    # Extract 4 new patches from the new grid
    patches = []
    for i in range(2):
        for j in range(2):
            patches.append(vertex_points[:, i * 3:(i + 1) * 3 + 1, j * 3:(j + 1) * 3 + 1])

    return torch.cat(patches, dim=0)


def evaluate_bspline_surface(control_points, u, v):
    """
    Evaluate the B-Spline surface point S(u, v) given the control points.

    Parameters:
        control_points (torch.Tensor): Tensor of shape (4, 4, 3) representing 4x4 control points in 3D.
        u (torch.Tensor): The u parameters in [0, 1] of shape (m,).
        v (torch.Tensor): The v parameters in [0, 1] of shape (n,).

    Returns:
        torch.Tensor: The surface points S(u, v) of shape (m, n, 3).
    """
    m, n = u.shape[0], v.shape[0]
    Bu = torch.stack([basis_functions(3, u_i) for u_i in u], dim=0)  # (m, 4)
    Bv = torch.stack([basis_functions(3, v_i) for v_i in v], dim=0)  # (n, 4)

    surface_points = torch.tensordot(Bu.to('cuda'), torch.tensordot(Bv.to('cuda'), control_points, dims=([1], [1])),
                                     dims=([1], [1]))
    return surface_points


def quaternion_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Compute the quaternion that rotates v1 to v2
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)

    w = torch.sqrt((1.0 + torch.sum(v1 * v2, dim=1)) / 2.0) + eps
    xyz = torch.cross(v1, v2) / (4.0 * w.unsqueeze(1) + eps)

    return torch.cat([w.unsqueeze(1), xyz], dim=1)



def b_spline_basis_function(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
        coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
        return coef1 * b_spline_basis_function(i, k - 1, t, knots) + coef2 * b_spline_basis_function(i + 1, k - 1, t,
                                                                                                     knots)

def b_spline_basis_functions_and_derivatives(degree, knots, t_values, device='cuda'):
    """
    Compute B-spline basis functions and their first and second derivatives for a given set of knots and parameter values.

    Parameters:
        degree (int): Degree of the B-spline basis functions.
        knots (torch.Tensor): Knot vector.
        t_values (torch.Tensor): Parameter values at which to evaluate the basis functions.

    Returns:
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the basis function values.
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the first derivative values.
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the second derivative values.
    """
    num_knots = len(knots)
    num_basis = num_knots - degree - 1
    num_t_values = len(t_values)

    # Initialize the table of basis function values
    N = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    dN = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    ddN = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)

    # Compute the zeroth-degree basis functions
    for i in range(num_knots - 1):
        N[:, i, 0] = ((knots[i] <= t_values) & (t_values < knots[i + 1])).float()

    # Compute the higher-degree basis functions iteratively
    for k in range(1, degree + 1):
        for i in range(num_knots - k - 1):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            coef1 = (t_values - knots[i]) / denom1 if denom1 != 0 else torch.zeros_like(t_values)
            coef2 = (knots[i + k + 1] - t_values) / denom2 if denom2 != 0 else torch.zeros_like(t_values)

            dcoef1 = 1.0 / denom1 if denom1 != 0 else torch.zeros_like(t_values)
            dcoef2 = -1.0 / denom2 if denom2 != 0 else torch.zeros_like(t_values)

            N[:, i, k] = coef1 * N[:, i, k - 1] + coef2 * N[:, i + 1, k - 1]
            dN[:, i, k] = (dcoef1 * N[:, i, k - 1] + coef1 * dN[:, i, k - 1] + dcoef2 * N[:, i + 1, k - 1]
                           + coef2 * dN[:,  i + 1,  k - 1])




            ddN[:, i, k] = dcoef1 * dN[:, i, k - 1] + coef1 * ddN[:, i, k - 1] + dcoef2 * dN[:, i + 1,
                                                                                          k - 1] + coef2 * ddN[:, i + 1,
                                                                                                           k - 1]

    return N[:, :num_basis, degree].to(device), dN[:, :num_basis, degree].to(device), ddN[:, :num_basis, degree].to(device)


def patch_subdivisions(xyz):
    """
    Given a BSpline patch (N patches in total), where each patch is defined by 4x4 3D control points,
    this function returns 4 new patches for each of the N patches.

    Original 4x4 grid:
    [A1, A2, A3, A4]
    [B1, B2, B3, B4]
    [C1, C2, C3, C4]
    [D1, D2, D3, D4]

    New 7x7 grid (0-based indices):
    [ A1,  M1,  A2,  M2,  A3,  M3,  A4 ]
    [ M4, CM1,  M5, CM2,  M6, CM3,  M7 ]
    [ B1,  M8,  B2,  M9,  B3, M10,  B4 ]
    [ M11,CM4, M12, CM5, M13, CM6, M14 ]
    [ C1, M15,  C2, M16,  C3, M17,  C4 ]
    [ M18,CM7, M19, CM8, M20, CM9, M21 ]
    [ D1, M22,  D2, M23,  D3, M24,  D4 ]

    Where:
    - `A1, A2, ..., D4` are original control points.
    - `M1, M2, ..., M24` are midpoints (either row or column).
    - `CM1, CM2, ..., CM9` are center midpoints.
    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing control points in 3D.

    Returns:
        torch.Tensor: Subdivided 4 new patches of the old patch according to its 3D control points tensor of shape (4N, 4, 4, 3).
    """
    N, H, W, C = xyz.shape


    # Compute the midpoints of the edges
    def compute_midpoints(tensor):
        return 0.5 * (tensor[:, :-1] + tensor[:, 1:])

    # Step 1: Compute midpoints for rows and columns
    rows_mid = compute_midpoints(xyz)
    cols_mid = compute_midpoints(xyz.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    # Step 2: Compute midpoints for the midpoints
    row_midpoints_mid = compute_midpoints(rows_mid)
    col_midpoints_mid = compute_midpoints(cols_mid.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    # Step 3: Compute center midpoints
    center_midpoints = 0.5 * (row_midpoints_mid + col_midpoints_mid)

    # Step 4: Assemble the new grid of control points
    new_grid = torch.zeros((xyz.shape[0], 7, 7, 3), dtype=xyz.dtype, device=xyz.device)
    new_grid[:, 0::2, 0::2] = xyz
    new_grid[:, 1::2, 0::2] = rows_mid
    new_grid[:, 0::2, 1::2] = cols_mid
    new_grid[:, 1::2, 1::2] = center_midpoints

    # Step 4: Extract the new patches
    patches = torch.cat((new_grid[:, 0:4, 0:4],
                        new_grid[:, 0:4, 3:],
                        new_grid[:, 3:, 0:4],
                        new_grid[:, 3:, 3:]), dim=0)

    return patches



def SH_interpolation(sh_features, target_shape=(7, 7)):
    """
    Interpolate SH features according to the interpolation of 3D points (xyz).

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing 3D points.
        sh_features (torch.Tensor): Tensor of shape (N, 4, 4, 25, 3) representing SH features.
        target_shape (tuple): The target shape (new_height, new_width) for interpolation.

    Returns:
        torch.Tensor: Interpolated SH features tensor of shape (N, new_height, new_width, 25, 3).
        torch.Tensor: Interpolated 3D points tensor of shape (N, new_height, new_width, 3).
    """
    # N, h, w, D = xyz.shape # Shape: (N, 3, 4, 4)
    num_patch, udim, vdim, num_coeffs, num_channels = sh_features.shape

    # Interpolate xyz
    # interpolated_xyz = F.interpolate(xyz.permute(0,3,1,2), size=(7,7), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    # Prepare sh_features for interpolation

    sh_features = sh_features.view(num_patch, udim, vdim, num_coeffs * num_channels)  # Shape: (N, 4, 4, 75)
    sh_features = sh_features.permute(0, 3, 1, 2)
    # Interpolate SH features
    interpolated_sh = F.interpolate(sh_features, size=(7, 7), mode='bilinear', align_corners=True)
    interpolated_sh = interpolated_sh.permute(0, 2, 3, 1)
    interpolated_sh = interpolated_sh.view(num_patch, target_shape[0], target_shape[1], num_coeffs,
                                           num_channels)  # Shape: (N, new_height, new_width, 25, 3)
    interpolated_sh = torch.stack([interpolated_sh[:, :4, :4, :],
                                   interpolated_sh[:, :4, 3:, :],
                                   interpolated_sh[:, 3:, :4, :],
                                   interpolated_sh[:, 3:, 3:, :]])

    return interpolated_sh.flatten(end_dim=1) #, interpolated_xyz.flatten(end_dim=1)


def normalize_point_cloud(points):
    """
    Normalize the point cloud to be bounded between -1 and 1.

    Parameters:
    points (torch.tensor): A numpy array of shape (N, 3) representing the point cloud.

    Returns:
    torch.tensor: The normalized point cloud.
    """

    true_shape = points.shape
    if not (points.shape[-1] == 3 and len(points.shape) == 2):
        points = points.reshape(-1, 3)


    # Find the minimum and maximum coordinates along each axis
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # uncomment if we want the scene be centered (Must attend also the cameras parameters)
    # center = (min_coords.values + max_coords.values) / 2. # Calculate the center of the bounding box

    # Calculate the scale factor
    scale = (max_coords.values - min_coords.values).max() / 2.

    # Normalize the points
    normalized_points = points / scale
    normalized_points = normalized_points.reshape(true_shape)

    return normalized_points

def analyze_gradient_trend_per_patch(gradients, window_size=32, top_k=5, tolerance=1e-12, device='cuda', param_group_name='', lr=0, epsilon=1e-6):
    # Aggregate gradients across all dimensions except BATCH and PATCHES
    if gradients.dim() == 5:  # Case A
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4))
    elif gradients.dim() == 6:  # Case B and C
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4, 5))
    else:
        raise ValueError("Unexpected gradient shape")

    BATCH, PATCHES = agg_gradients.shape

    # Compute the moving average
    cumsum = torch.cumsum(agg_gradients, dim=0)
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Compute the trend
    X = torch.arange(window_size, BATCH, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, PATCHES)
    y = moving_avg

    # Compute the slope using least squares method
    X_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)

    numerator = torch.sum((X - X_mean) * (y - y_mean), dim=0)
    denominator = torch.sum((X - X_mean) ** 2, dim=0) + epsilon

    slope = numerator / denominator

    tolerance = slope.std() * 0.1
    print(f"Tolerance is {tolerance}")

    # Determine convergence or divergence
    stable = (slope.abs() <= tolerance)

    convergence = (slope < 0) & ~stable
    divergence = (slope > 0) & ~stable

    # Select diverging or stable patches
    non_converging = divergence | stable
    non_converging_slopes = slope[non_converging]

    # Get top_k elements
    top_k_mask = torch.zeros_like(slope, dtype=torch.bool)
    if non_converging_slopes.numel() > 0:
        top_k = min(top_k, non_converging_slopes.numel())
        _, top_k_indices = torch.topk(non_converging_slopes, k=top_k)

        # Create boolean mask for top_k slopes
        non_converging_indices = torch.nonzero(non_converging).squeeze()
        if non_converging_indices.ndim == 0:
            top_k_mask[non_converging_indices] = True
        else:
            top_k_mask[non_converging_indices[top_k_indices]] = True

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_all, ax_diverging, ax_converging, ax_stable = axes.flatten()

    for patch in range(PATCHES):
        ax_all.plot(moving_avg[:, patch].cpu(), label=f'Patch {patch}' if top_k_mask[patch].item() else None, alpha=0.5)
        if divergence[patch]:
            ax_diverging.plot(moving_avg[:, patch].cpu(), label=f'Patch {patch}', alpha=0.5, color='red')
        if convergence[patch]:
            ax_converging.plot(moving_avg[:, patch].cpu(), label=f'Patch {patch}', alpha=0.5, color='green')
        if stable[patch]:
            ax_stable.plot(moving_avg[:, patch].cpu(), label=f'Patch {patch}', alpha=0.5, color='blue')
    # ax.set_title(f'Gradient Trend Analysis per Patch for {param_group_name}, and init LR: {lr}')
    # ax.set_xlabel('Batch Index')
    # ax.set_ylabel(f'Gradient Moving Average (window size: {window_size})')
    # ax.legend(loc='upper right')

    ax_all.set_title(f'Gradient Trend Analysis per Patch for {param_group_name}, and init LR: {lr}')
    ax_all.set_xlabel('Batch Index')
    ax_all.set_ylabel(f'Gradient Moving Average (window size: {window_size})')
    ax_all.legend(loc='upper right')

    ax_diverging.set_title('Diverging Patches')
    ax_diverging.set_xlabel('Batch Index')
    ax_diverging.set_ylabel('Gradient Moving Average')

    ax_converging.set_title('Converging Patches')
    ax_converging.set_xlabel('Batch Index')
    ax_converging.set_ylabel('Gradient Moving Average')

    ax_stable.set_title('Stable Patches')
    ax_stable.set_xlabel('Batch Index')
    ax_stable.set_ylabel('Gradient Moving Average')

    plt.tight_layout()
    plt.show()
    # # Log to wandb
    # wandb.log({
    #     "converging": wandb.Histogram(convergence.cpu().numpy()),
    #     "diverging": wandb.Histogram(divergence.cpu().numpy()),
    #     "stable": wandb.Histogram(stable.cpu().numpy()),
    #     "non_converging": wandb.Histogram(non_converging.cpu().numpy()),
    #     "slope": wandb.Histogram(slope.cpu().numpy()),
    #     "top_k_mask": wandb.Histogram(top_k_mask.cpu().numpy())
    # })
    return {
        "converging": convergence,
        "diverging": divergence,
        "stable": stable,
        "non_converging": non_converging,
        "slope": slope,
        "top_k_mask": top_k_mask
    }


def analyze_gradient_trend_per_patch1(gradients, window_size=128, top_k=5, device='cuda', param_group_name='', lr=0):
    # Aggregate gradients across all dimensions except BATCH and PATCHES
    if gradients.dim() == 5:  # Case A
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4))
    elif gradients.dim() == 6:  # Case B and C
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4, 5))
    else:
        raise ValueError("Unexpected gradient shape")

    BATCH, PATCHES = agg_gradients.shape

    # Compute the moving average
    cumsum = torch.cumsum(agg_gradients, dim=0)
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Compute the trend
    X = torch.arange(window_size, BATCH, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, PATCHES)
    y = moving_avg

    # Compute the slope using least squares method
    X_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)

    numerator = torch.sum((X - X_mean) * (y - y_mean), dim=0)
    denominator = torch.sum((X - X_mean) ** 2, dim=0) + 1e-8  # Regularization term added

    slope = numerator / denominator

    # Determine convergence or divergence
    convergence = slope < 0
    divergence = slope > 0
    stable = slope == 0

    # Select diverging or stable patches
    non_converging = divergence | stable
    non_converging_slopes = slope[non_converging]

    # Get top_k elements
    top_k_mask = torch.zeros_like(slope, dtype=torch.bool)
    if non_converging_slopes.numel() > 0:
        top_k = min(top_k, non_converging_slopes.numel())
        _, top_k_indices = torch.topk(non_converging_slopes, k=top_k)

        # Create boolean mask for top_k slopes
        non_converging_indices = torch.nonzero(non_converging).squeeze()
        if non_converging_indices.ndim == 0:
            top_k_mask[non_converging_indices] = True
        else:
            top_k_mask[non_converging_indices[top_k_indices]] = True

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    for patch in range(PATCHES):
        ax.plot(moving_avg[:, patch].cpu(), label=f'Patch {patch}' if top_k_mask[patch].item() else None, alpha=0.5)

    converging_patches = torch.nonzero(convergence).squeeze().cpu().tolist()
    diverging_patches = torch.nonzero(divergence).squeeze().cpu().tolist()
    stable_patches = torch.nonzero(stable).squeeze().cpu().tolist()

    if converging_patches:
        ax.plot([], [], label='Converging', color='green')
    if diverging_patches:
        ax.plot([], [], label='Diverging', color='red')
    if stable_patches:
        ax.plot([], [], label='Stable', color='blue')

    ax.set_title(f'Gradient Trend Analysis per Patch for {param_group_name}, and init LR: {lr}')
    ax.set_xlabel('Batch Index')
    ax.set_ylabel(f'Gradient Moving Average (window size: {window_size})')
    ax.legend(loc='upper right')
    plt.show()

    return {
        "converging": convergence,
        "diverging": divergence,
        "stable": stable,
        "non_converging": non_converging,
        "slope": slope,
        "top_k_mask": top_k_mask
    }



def analyze_gradient_trend_per_patch2(gradients, window_size=64, top_k=5, device='cuda'):
    # Aggregate gradients across all dimensions except BATCH and PATCHES
    if gradients.dim() == 5:  # Case A
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4))
    elif gradients.dim() == 6:  # Case B and C
        agg_gradients = gradients.abs().mean(dim=(2, 3, 4, 5))
    else:
        raise ValueError("Unexpected gradient shape")

    BATCH, PATCHES = agg_gradients.shape

    # Compute the moving average
    cumsum = torch.cumsum(agg_gradients, dim=0)
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Compute the trend
    X = torch.arange(window_size, BATCH, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, PATCHES)
    y = moving_avg

    # Compute the slope using least squares method
    X_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)

    numerator = torch.sum((X - X_mean) * (y - y_mean), dim=0)
    denominator = torch.sum((X - X_mean) ** 2, dim=0)

    slope = numerator / denominator

    # Determine convergence or divergence
    convergence = slope < 0
    divergence = slope > 0
    stable = slope == 0

    # Select diverging or stable patches
    non_converging = divergence | stable
    non_converging_slopes = slope[non_converging]

    # Get top_k elements
    top_k_mask = torch.zeros_like(slope, dtype=torch.bool)
    if non_converging_slopes.numel() > 0:
        top_k = min(top_k, non_converging_slopes.numel())
        _, top_k_indices = torch.topk(non_converging_slopes, k=top_k)

        # Create boolean mask for top_k slopes
        non_converging_indices = torch.nonzero(non_converging).squeeze()
        if non_converging_indices.ndim == 0 :
            top_k_mask[non_converging_indices] = True
        else:
            top_k_mask[non_converging_indices[top_k_indices]] = True

    return {
        "converging": convergence,
        "diverging": divergence,
        "stable": stable,
        "non_converging": non_converging,
        "slope": slope,
        "top_k_mask": top_k_mask
    }

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class Gaussians(NamedTuple):
    xyz: torch.Tensor
    features: torch.Tensor
    scaling: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor
    active_sh_degree: int = 0


class SplineSurface(NamedTuple):
    surface_points: torch.Tensor
    du: torch.Tensor
    dv: torch.Tensor
    dduv: torch.Tensor
    dduu: torch.Tensor
    ddvv: torch.Tensor
    curvature_gaussian: torch.Tensor
    curvature_mean: torch.Tensor
