import torch
from typing import NamedTuple
from matplotlib import pyplot as plt
import torch.nn.functional as F


def interpolate_SH_single_patch(sh_values, sh_degree, num_samples_u, num_samples_v, device='cuda'):
    """
    Interpolates spherical harmonics values for sampled points on a single patch.

    Parameters:
        control_points (torch.Tensor): Tensor of shape (4, 4, 3) representing the control points of the surface patch.
        sh_values (torch.Tensor): Tensor of shape (4, 4, 3, (max_SH_degree + 1) ** 2) representing the SH values at the control points.
        num_samples_u (int): Number of samples in the u direction.
        num_samples_v (int): Number of samples in the v direction.

    Returns:
        torch.Tensor: Interpolated SH values of shape (num_samples_u, num_samples_v, 3, (max_SH_degree + 1) ** 2).
    """
    # Create sampling grid
    u = torch.linspace(0, 1, num_samples_u, device=device)
    v = torch.linspace(0, 1, num_samples_v, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    grid = torch.stack((u_grid, v_grid), dim=-1)  # Shape: (num_samples_u, num_samples_v, 2)
    grid = grid.unsqueeze(0)  # Add batch dimension

    # Normalize the grid to the range [-1, 1] for grid_sample
    grid = 2.0 * grid - 1.0

    # Reshape the SH values to (3 * (max_SH_degree + 1) ** 2, 4, 4) for grid_sample
    sh_values_reshaped = sh_values.reshape(3 * (sh_degree + 1) ** 2, 4, 4)

    # Use grid_sample for bilinear interpolation
    interpolated_sh_values = F.grid_sample(sh_values_reshaped.unsqueeze(0), grid, mode='bilinear', align_corners=True)

    # Reshape the interpolated SH values back to the original format
    interpolated_sh_values = interpolated_sh_values.reshape(3*(sh_degree+1)**2, num_samples_u, num_samples_v)
    interpolated_sh_values = interpolated_sh_values.T.reshape(-1, (sh_degree+1)**2, 3)  # Shape: (num_samples_u, num_samples_v, 3, sh_dim)

    return torch.clamp(interpolated_sh_values, 0.0, 1.0)

def interpolate_SH(sh_values_batch, num_samples_u=10, num_samples_v=10, device='cuda'):
    """
    Interpolates spherical harmonics values for sampled points on a batch of surface patches.

    Parameters:
        sh_values_batch (torch.Tensor): Tensor of shape (M * 4 * 4, 3, (max_SH_degree + 1) ** 2) representing the SH values at the control points.
        num_samples_u (int): Number of samples in the u direction.
        num_samples_v (int): Number of samples in the v direction.

    Returns:
        torch.Tensor: Interpolated SH values of shape (M, num_samples_u, num_samples_v, 3, (max_SH_degree + 1) ** 2).
    """
    M, SH_deg, _ = sh_values_batch.shape
    SH_deg = int(SH_deg ** .5 - 1)
    # Normalize the control points to the range [0, 1]
    u = torch.linspace(0, 1, num_samples_u, device=device, requires_grad=True)
    v = torch.linspace(0, 1, num_samples_v, device=device, requires_grad=True)

    # Create a meshgrid for sampling
    u_grid, v_grid = torch.meshgrid(u, v)
    grid = torch.stack((u_grid.to(device), v_grid.to(device)), dim=-1)  # Shape: (num_samples_u, num_samples_v, 2)
    grid = grid.expand(M, -1, -1, -1) # Shape: (M, num_samples_u, num_samples_v, 2)

    # Normalize the grid to the range [-1, 1] for grid_sample
    grid = 2.0 * grid - 1.0

    # Reshape the SH values to (M, 3 * (max_SH_degree + 1) ** 2, 4, 4) for grid_sample
    sh_values_batch = sh_values_batch.view(M, 3 * (int(SH_deg) + 1) ** 2)

    # M, _, _, _, sh_dim = sh_values_batch.shape
    # sh_values_reshaped = sh_values_batch.permute(0, 4, 1, 2, 3).reshape(M, -1, 4, 4)

    # Use grid_sample for bilinear interpolation
    interpolated_sh_values = F.grid_sample(sh_values_batch.view(32, 16, -1, 4, 4), grid.reshape(32, 16, 10, 10, 2),
                                           mode='bilinear', align_corners=True)

    # Reshape the interpolated SH values back to the original format
    interpolated_sh_values= interpolated_sh_values.view(32 * num_samples_u * num_samples_v, int(SH_deg + 1) ** 2, 3)

    return torch.clamp(interpolated_sh_values, 0.0, 1.0)


def orthogonalize_and_normalize_batch(u, v, epsilon=1e-6):
    # Normalize u
    u = u / (torch.norm(u, dim=1, keepdim=True) + epsilon)

    # Orthogonalize v with respect to u
    dot_product = torch.sum(v * u, dim=1, keepdim=True)
    v = v - dot_product * u

    # Normalize v
    v = v / (torch.norm(v, dim=1, keepdim=True) + epsilon)

    return u, v


def rotation_matrix_to_quaternion_batch(R):
    # Convert rotation matrix to quaternion for batch
    batch_size = R.shape[0]
    K = torch.zeros((batch_size, 4, 4), device=R.device)

    K[:, 0, 0] = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 3.0
    K[:, 1, 1] = (R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) / 3.0
    K[:, 2, 2] = (-R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]) / 3.0
    K[:, 3, 3] = (-R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]) / 3.0
    K[:, 0, 1] = K[:, 1, 0] = (R[:, 1, 2] - R[:, 2, 1]) / 3.0
    K[:, 0, 2] = K[:, 2, 0] = (R[:, 2, 0] - R[:, 0, 2]) / 3.0
    K[:, 0, 3] = K[:, 3, 0] = (R[:, 0, 1] - R[:, 1, 0]) / 3.0
    K[:, 1, 2] = K[:, 2, 1] = (R[:, 0, 1] + R[:, 1, 0]) / 3.0
    K[:, 1, 3] = K[:, 3, 1] = (R[:, 2, 0] + R[:, 0, 2]) / 3.0
    K[:, 2, 3] = K[:, 3, 2] = (R[:, 1, 2] + R[:, 2, 1]) / 3.0

    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = torch.nan_to_num(eigenvalues, nan=0.0, neginf=0.0, posinf=0.0)
    eigenvectors = torch.nan_to_num(eigenvectors, nan=0.0, neginf=0.0, posinf=0.0)
    q = eigenvectors[:, :, torch.argmax(eigenvalues, dim=1)]
    try:
        q = q[:, :, 0]
    except IndexError as e:
        print(e)

    return q


def quaternion_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Compute the quaternion that rotates v1 to v2
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)

    w = torch.sqrt((1.0 + torch.sum(v1 * v2, dim=1)) / 2.0) + eps
    xyz = torch.cross(v1, v2) / (4.0 * w.unsqueeze(1) + eps)

    return torch.cat([w.unsqueeze(1), xyz], dim=1)

def to_quaternions(u, v, adjuster=torch.ones((1,4), device='cuda'), epsilon=1e-6):
    # Ensure u and v are orthogonal and normalized
    u, v = orthogonalize_and_normalize_batch(u, v, epsilon)

    # Compute the third orthogonal vector
    w = torch.cross(u, v, dim=1)

    # Construct the rotation matrix
    R = torch.stack((u, v, w), dim=2)

    # Convert the rotation matrix to a quaternion
    q = rotation_matrix_to_quaternion_batch(R)

    return q.detach() * adjuster


def b_spline_basis_function(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
        coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
        return coef1 * b_spline_basis_function(i, k - 1, t, knots) + coef2 * b_spline_basis_function(i + 1, k - 1, t,
                                                                                                     knots)


def b_spline_basis_function_derivative(i, k, t, knots):
    if k == 0:
        return 0.0
    else:
        coef1 = k / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
        coef2 = k / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
        return coef1 * b_spline_basis_function(i, k - 1, t, knots) - coef2 * b_spline_basis_function(i + 1, k - 1, t,
                                                                                                     knots)


def b_spline_basis_functions(degree, knots, t_values):
    """
    Compute B-spline basis functions for a given set of knots and parameter values.

    Parameters:
        degree (int): Degree of the B-spline basis functions.
        knots (torch.Tensor): Knot vector.
        t_values (torch.Tensor): Parameter values at which to evaluate the basis functions.

    Returns:
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the basis function values.
    """
    num_knots = len(knots)
    num_basis = num_knots - degree - 1
    num_t_values = len(t_values)

    # Initialize the table of basis function values
    N = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)

    # Compute the zeroth-degree basis functions
    for i in range(num_knots - 1):
        N[:, i, 0] = ((knots[i] <= t_values) & (t_values < knots[i + 1])).float()

    # Compute the higher-degree basis functions iteratively
    for k in range(1, degree + 1):
        for i in range(num_knots - k - 1):
            coef1 = ((t_values - knots[i]) / (knots[i + k] - knots[i])).unsqueeze(-1)
            coef2 = ((knots[i + k + 1] - t_values) / (knots[i + k + 1] - knots[i + 1])).unsqueeze(-1)

            coef1[knots[i + k] == knots[i]] = 0
            coef2[knots[i + k + 1] == knots[i + 1]] = 0

            N[:, i, k] = coef1.squeeze() * N[:, i, k - 1] + coef2.squeeze() * N[:, i + 1, k - 1]

    return N[:, :num_basis, degree]


def compute_all_basis_functions_and_derivatives(degree, knots, t_values):
    """
    Compute B-spline basis functions and their first and second derivatives for all patches.

    Parameters:
        degree (int): Degree of the B-spline basis functions.
        knots (torch.Tensor): Knot vectors of shape (num_patches, num_knots).
        t_values (torch.Tensor): Parameter values at which to evaluate the basis functions.

    Returns:
        torch.Tensor: Tensor of shape (num_patches, len(t_values), num_basis) containing the basis function values.
        torch.Tensor: Tensor of shape (num_patches, len(t_values), num_basis) containing the first derivative values.
        torch.Tensor: Tensor of shape (num_patches, len(t_values), num_basis) containing the second derivative values.
    """
    num_patches, num_knots = knots.shape
    num_basis = num_knots - degree - 1
    num_t_values = len(t_values)

    # Initialize the table of basis function values
    N = torch.zeros((num_patches, num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    dN = torch.zeros((num_patches, num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    ddN = torch.zeros((num_patches, num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)

    # Compute the zeroth-degree basis functions
    for i in range(num_knots - 1):
        N[:, :, i, 0] = ((knots[:, i:i + 1] <= t_values) & (t_values < knots[:, i + 1:i + 2])).float()

    # Compute the higher-degree basis functions iteratively
    for k in range(1, degree + 1):
        for i in range(num_knots - k - 1):
            denom1 = knots[:, i + k:i + k + 1] - knots[:, i:i + 1]
            denom2 = knots[:, i + k + 1:i + k + 2] - knots[:, i + 1:i + 2]

            coef1 = ((t_values - knots[:, i:i + 1]) / denom1).unsqueeze(-1)
            coef2 = ((knots[:, i + k + 1:i + k + 2] - t_values) / denom2).unsqueeze(-1)

            dcoef1 = (1.0 / denom1).unsqueeze(-1)
            dcoef2 = (-1.0 / denom2).unsqueeze(-1)

            coef1[denom1 == 0] = 0
            coef2[denom2 == 0] = 0
            dcoef1[denom1 == 0] = 0
            dcoef2[denom2 == 0] = 0

            N[:, :, i, k] = coef1.squeeze() * N[:, :, i, k - 1] + coef2.squeeze() * N[:, :, i + 1, k - 1]
            dN[:, :, i, k] = dcoef1.squeeze() * N[:, :, i, k - 1] + coef1.squeeze() * dN[:, :, i,
                                                                                      k - 1] + dcoef2.squeeze() * N[:,
                                                                                                                  :,
                                                                                                                  i + 1,
                                                                                                                  k - 1] + coef2.squeeze() * dN[
                                                                                                                                             :,
                                                                                                                                             :,
                                                                                                                                             i + 1,
                                                                                                                                             k - 1]

            ddN[:, :, i, k] = dcoef1.squeeze() * dN[:, :, i, k - 1] + coef1.squeeze() * ddN[:, :, i,
                                                                                        k - 1] + dcoef2.squeeze() * dN[
                                                                                                                    :,
                                                                                                                    :,
                                                                                                                    i + 1,
                                                                                                                    k - 1] + coef2.squeeze() * ddN[
                                                                                                                                               :,
                                                                                                                                               :,
                                                                                                                                               i + 1,
                                                                                                                                               k - 1]

    return N[:, :, :num_basis, degree], dN[:, :, :num_basis, degree], ddN[:, :, :num_basis, degree]


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


def patch_subdivision(xyz, target_shape=(7, 7)):
    """
    Given BSpline patches (N patches in total), where each patch is defined by 4x4 3D control points,
    this function returns 4 new patches for each of the N patches.

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing control points in 3D.
    Returns:
        torch.Tensor: Subdivided 4 new patches of the old patch according to its 3D control points
                      tensor of shape (4N, 4, 4, 3).
    """
    N = xyz.shape[0]
    device = xyz.device

    # Define subdivision matrix
    S = torch.tensor([
        [1/2, 1/2, 0, 0],
        [1/4, 3/4, 0, 0],
        [0, 1, 0, 0],
        [0, 3/4, 1/4, 0],
        [0, 1/2, 1/2, 0],
        [0, 1/4, 3/4, 0],
        [0, 0, 1, 0]
    ], device=device)

    # Perform subdivision
    u_subdivided = torch.einsum('ij,bjkl->bikl', S, xyz)
    uv_subdivided = torch.einsum('ij,bkjl->bkil', S, u_subdivided)
    return torch.cat([
        uv_subdivided[:, :4, :4],  # Top-left
        uv_subdivided[:, :4, 3:7],  # Top-right
        uv_subdivided[:, 3:7, :4], # Bottom-left
        uv_subdivided[:, 3:7, 3:7]  # Bottom-right
    ])


def patch_subdivisions(xyz):
    """
    Given a BSpline patch (N patches in total), where each patch is defined by 4x4 3D control points,
    this function returns 4 new patches for each of the N patches.

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing control points in 3D.

    Returns:
        torch.Tensor: Subdivided 4 new patches of the old patch according to its 3D control points tensor of shape (4N, 4, 4, 3).
    """

    # Compute the midpoints of the edges
    def compute_midpoints(tensor):
        return 0.5 * (tensor[:, :-1] + tensor[:, 1:])

    # Step 1: Compute the new rows and columns
    xyz_new_rows = compute_midpoints(xyz)
    xyz_new_cols = compute_midpoints(xyz.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    # Step 2: Compute the midpoints of the new rows and columns
    xyz_midpoints = compute_midpoints(xyz_new_rows.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    # Step 3: Compute the new control points
    xyz_new = torch.zeros((xyz.shape[0], 7, 7, xyz.shape[-1]), dtype=xyz.dtype, device=xyz.device)
    xyz_new[:, 0::2, 0::2] = xyz
    xyz_new[:, 1::2, 0::2] = xyz_new_rows
    xyz_new[:, 0::2, 1::2] = xyz_new_cols
    xyz_new[:, 1::2, 1::2] = xyz_midpoints

    # Step 4: Extract the new patches
    patches = torch.cat((xyz_new[:, 0:4, 0:4],
                        xyz_new[:, 0:4, 3:],
                        xyz_new[:, 3:, 0:4],
                        xyz_new[:, 3:, 3:]), dim=0)

    return patches


def interpolate_sh_coefficients(sh_coeffs):
    """
    Interpolates SH coefficients tensor to the target shape using bilinear interpolation.

    Parameters:
        sh_coeffs (torch.Tensor): Input tensor of SH coefficients of shape
                                  (num_patches * height * width, (max_sh_degree + 1) ** 2, 3).
        target_shape (tuple): The target shape (new_height, new_width) for interpolation.

    Returns:
        torch.Tensor: Interpolated SH coefficients tensor.
    """
    # Reshape to (N, C, H, W) format
    num_patches, H, W, num_coeffs, num_channels = sh_coeffs.shape
    sh_coeffs1 = sh_coeffs.reshape(num_patches, H, W, num_channels, num_coeffs).unsqueeze(0)
    # sh_coeffs1 = sh_coeffs.permute(0, 3,1,2)
    D = num_channels * num_coeffs
    # Perform interpolation
    sh_coeffs_interp = F.interpolate(sh_coeffs1, size=(7, 7, num_channels, num_coeffs), mode='trilinear', align_corners=True).squeeze(0).reshape(num_patches, H*2-1, W*2-1,num_coeffs, num_channels)
    sh_coeffs_interp = torch.stack([sh_coeffs_interp[:, :4,:4, ...], sh_coeffs_interp[:, 3:,3:, ...], sh_coeffs_interp[:, :4, 3:, ...], sh_coeffs_interp[:, 3:,:4, ...]])
    return sh_coeffs_interp.flatten(end_dim=3)


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


def compute_scale_factors(dU, dV, UxV):
    """
    Compute the scale factors in the x, y, and z directions from directional derivatives dU and dV.

    Parameters:
        dU (torch.Tensor): Directional derivative in the u direction. Shape (N, 3).
        dV (torch.Tensor): Directional derivative in the v direction. Shape (N, 3).

    Returns:
        torch.Tensor: Scale factors in x, y, z directions. Shape (N, 3).
    """
    # Compute the normal vector
    N = torch.cross(dU, dV, dim=1)

    # Compute the norms of the tangent vectors
    norm_Tu = torch.norm(dU, dim=1, keepdim=True)
    norm_Tv = torch.norm(dV, dim=1, keepdim=True)
    norm_N = torch.norm(N, dim=1, keepdim=True)

    # Compute the scale factors (norms of the tangent vectors)
    scale_factors = torch.cat((norm_Tu, norm_Tv, norm_N), dim=1)

    return scale_factors

def compute_scale_factors(dU, dV, nUxV=None, eps=1e-6):
    # Compute the magnitude of the surface normal
    if nUxV is None:
        nUxV = torch.norm(torch.cross(dU, dV, dim=-1), dim=-1, keepdim=True)

    # Compute the magnitude of the first fundamental form (E, G)
    E = torch.sum(dU * dU, dim=-1)
    G = torch.sum(dV * dV, dim=-1)

    # Compute the magnitude of the second fundamental form (L, N)
    L = torch.sum(dU * dU, dim=-1)
    N = torch.sum(dV * dV, dim=-1)

    # Compute the mean curvature
    curvature_mean = (E * N + G * L - 2 * torch.sum(dU * dV, dim=-1)) / (2 * (E * G - torch.sum(dU * dV, dim=-1)**2) + eps)

    # Compute the Gaussian curvature
    curvature_gaussian = (L * N - torch.sum(dU * dV, dim=-1)**2) / (E * G - torch.sum(dU * dV, dim=-1)**2 + eps)

    # Apply a sigmoid function to the mean curvature
    curvature_mean = torch.log(curvature_mean)

    # Apply a logarithmic transformation to the Gaussian curvature
    curvature_gaussian = torch.log(curvature_gaussian + 1e-9)

    # Combine the transformed curvatures
    combined_curvatures = curvature_mean + curvature_gaussian

    # Normalize the combined curvatures
    combined_curvatures = (combined_curvatures - combined_curvatures.min()) / (combined_curvatures.max() - combined_curvatures.min() + eps)

    # Apply a scaling factor to the combined curvatures
    scaling_factors = combined_curvatures * nUxV.squeeze(1)

    return scaling_factors
def catmull_rom_spline(p0, p1, p2, p3, num_points=100):
    t = torch.linspace(0, 1, num_points, device=p0.device)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom spline basis matrix
    m = torch.tensor([[0, 1, 0, 0],
                      [-0.5, 0, 0.5, 0],
                      [1, -2.5, 2, -0.5],
                      [-0.5, 1.5, -1.5, 0.5]], device=p0.device)

    # Points matrix
    p = torch.stack([p0, p1, p2, p3])

    # Basis functions
    T = torch.stack([t3, t2, t, torch.ones_like(t)], dim=1)

    # Compute the spline points
    spline_points = torch.mm(T, torch.mm(m, p))

    return spline_points


def compute_quaternion_from_normals(dus, dvs):
    """
    Computes the quaternion representing the orientation of the Gaussian.

    Parameters:
        dus (torch.Tensor): First derivatives with respect to u (shape: BATCH_SIZE x res x res x 3).
        dvs (torch.Tensor): First derivatives with respect to v (shape: BATCH_SIZE x res x res x 3).

    Returns:
        torch.Tensor: Quaternion values (shape: BATCH_SIZE x res x res x 4).
    """
    normals = torch.cross(dus, dvs, dim=-1)
    normals = F.normalize(normals, p=2, dim=-1)

    # Create a quaternion that aligns the z-axis with the normal
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=normals.device).expand_as(normals)
    dot_product = (z_axis * normals).sum(dim=-1, keepdim=True)
    cross_product = torch.cross(z_axis, normals, dim=-1)

    q_w = torch.sqrt((1.0 + dot_product) / 2.0)
    q_xyz = cross_product / (2.0 * q_w)

    quaternion = torch.cat([q_w, q_xyz], dim=-1)
    return quaternion





import math
from torch.optim.lr_scheduler import _LRScheduler


class DecayingCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, alpha=0.5, beta=0.5,
                 verbose=False, start_decay_cycle=0, stop_decay_cycle=float('inf')):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0: First cycle step size.
            T_mult: Cycle steps magnification. Default: 1.
            eta_min: Minimum learning rate. Default: 0.
            last_epoch: The index of last epoch. Default: -1.
            alpha: Decay factor for max_lr after each restart. Default: 0.5.
            beta: Decay factor for min_lr after each restart. Default: 0.5.
            verbose: If True, prints a message to stdout for each update. Default: False.
            start_decay_cycle: Cycle number to start decay. Default: 0 (start immediately).
            stop_decay_cycle: Cycle number to stop decay. Default: inf (never stop).
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not 0 <= alpha < 1:
            raise ValueError(f"Expected 0 <= alpha < 1, but got {alpha}")
        if not 0 <= beta < 1:
            raise ValueError(f"Expected 0 <= beta < 1, but got {beta}")
        if not isinstance(start_decay_cycle, int) or start_decay_cycle < 0:
            raise ValueError(f"Expected non-negative integer start_decay_cycle, but got {start_decay_cycle}")
        if not (isinstance(stop_decay_cycle, (int, float)) and stop_decay_cycle > start_decay_cycle):
            raise ValueError(f"Expected stop_decay_cycle > start_decay_cycle, but got {stop_decay_cycle}")

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.base_eta_min = eta_min
        self.alpha = alpha
        self.beta = beta
        self.cycle = 0
        self.T_cur = last_epoch
        self.verbose = verbose
        self.start_decay_cycle = start_decay_cycle
        self.stop_decay_cycle = stop_decay_cycle

        self.base_max_lrs = [group['lr'] for group in optimizer.param_groups]
        self.eta_min = self.base_eta_min
        self.base_lrs = [lr for lr in self.base_max_lrs]
        super(DecayingCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur == 0:
            return self.calculate_lr()
        elif (self.T_cur + 1) % self.T_i == 0:
            return self.calculate_lr(is_restart=True)
        else:
            return self.calculate_lr()

    def calculate_lr(self, is_restart=False):
        cycle_factor = 1
        if self.start_decay_cycle <= self.cycle < self.stop_decay_cycle:
            cycle_factor = self.alpha ** (self.cycle - self.start_decay_cycle + int(is_restart))

        return [
            self.eta_min + (max_lr * cycle_factor - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for max_lr in self.base_max_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        if self.start_decay_cycle <= self.cycle < self.stop_decay_cycle:
            self.eta_min = self.base_eta_min * (self.beta ** (self.cycle - self.start_decay_cycle))
        else:
            self.eta_min = self.base_eta_min

        self.last_epoch = math.floor(epoch)

        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

        if self.verbose:
            print(f'Epoch {self.last_epoch}: adjusting learning rate to {new_lrs}')


def plot_lr_schedule(scheduler, num_epochs):
    """
    Utility function to plot the learning rate schedule.

    Args:
        scheduler (DecayingCosineAnnealingWarmRestarts): The scheduler to validate.
        num_epochs (int): Number of epochs to simulate.

    Returns:
        None: Displays a plot of the learning rate schedule.
    """
    lr_history = []
    for epoch in range(num_epochs):
        current_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
        lr_history.append(current_lrs)
        scheduler.step()

    epochs = list(range(num_epochs))
    min_lrs = [min(lrs) for lrs in lr_history]
    max_lrs = [max(lrs) for lrs in lr_history]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, min_lrs, label='Min LR', color='blue')
    plt.plot(epochs, max_lrs, label='Max LR', color='red')
    plt.fill_between(epochs, min_lrs, max_lrs, alpha=0.2, color='gray')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()


def analyze_gradient_trend_per_patch(gradients, window_size=5, top_k=5, device='cuda'):
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
        "slope": slope,
        "top_k_mask": top_k_mask
    }
def analyze_gradient_trend(gradients, window_size=5, device='cuda'):
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
    convergence = torch.where(slope < 0, True, False)
    divergence = torch.where(slope > 0, True, False)
    stable = torch.where(slope == 0, True, False)

    return {
        "converging": convergence,
        "diverging": divergence,
        "stable": stable,
        "slope": slope
    }
def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def generate_knot_vector(num_control_points, degree):
    """
    Generate an open uniform knot vector for B-spline.

    Parameters:
        num_control_points (int): Number of control points.
        degree (int): Degree of the B-spline.

    Returns:
        torch.Tensor: Knot vector.
    """
    num_knots = num_control_points + degree + 1
    knot_vector = torch.zeros(num_knots, dtype=torch.float32)

    for i in range(1, num_knots - degree):
        knot_vector[i + degree - 1] = i

    knot_vector[-degree:] = num_knots - degree - 1
    knot_vector /= (num_knots - degree - 1)

    return knot_vector
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
