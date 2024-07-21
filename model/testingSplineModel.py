import torch

def compute_spline_curvature_iterative(control_points, degree_u, degree_v, knots_u, knots_v, num_points=100):
    def b_spline_basis_function(i, k, t, knots):
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
        else:
            coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
            coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[
                i + 1] else 0
            return coef1 * b_spline_basis_function(i, k - 1, t, knots) + coef2 * b_spline_basis_function(i + 1, k - 1,
                                                                                                         t, knots)

    def b_spline_basis_function_derivative(i, k, t, knots):
        if k == 0:
            return 0.0
        else:
            coef1 = k / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
            coef2 = k / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
            return coef1 * b_spline_basis_function(i, k - 1, t, knots) - coef2 * b_spline_basis_function(i + 1, k - 1,
                                                                                                         t, knots)

    u_values = torch.linspace(knots_u[degree_u], knots_u[-degree_u - 1], num_points)
    v_values = torch.linspace(knots_v[degree_v], knots_v[-degree_v - 1], num_points)

    control_points_tensor = torch.tensor(control_points, dtype=torch.float32)

    b_u_tensor = torch.tensor(
        [[b_spline_basis_function(k, degree_u, u, knots_u) for k in range(len(control_points))] for u in u_values],
        dtype=torch.float32)
    b_v_tensor = torch.tensor(
        [[b_spline_basis_function(l, degree_v, v, knots_v) for l in range(len(control_points[0]))] for v in v_values],
        dtype=torch.float32)

    db_u_tensor = torch.tensor(
        [[b_spline_basis_function_derivative(k, degree_u, u, knots_u) for k in range(len(control_points))] for u in
         u_values], dtype=torch.float32)
    db_v_tensor = torch.tensor(
        [[b_spline_basis_function_derivative(l, degree_v, v, knots_v) for l in range(len(control_points[0]))] for v in
         v_values], dtype=torch.float32)

    ddb_u_tensor = torch.tensor(
        [[b_spline_basis_function_derivative(k, degree_u - 1, u, knots_u) for k in range(len(control_points))] for u in
         u_values], dtype=torch.float32)
    ddb_v_tensor = torch.tensor(
        [[b_spline_basis_function_derivative(l, degree_v - 1, v, knots_v) for l in range(len(control_points[0]))] for v
         in v_values], dtype=torch.float32)

    surface_points = torch.einsum('ik,jl,klm->ijm', b_u_tensor, b_v_tensor, control_points_tensor)
    derivative_u = torch.einsum('ik,jl,klm->ijm', db_u_tensor, b_v_tensor, control_points_tensor)
    derivative_v = torch.einsum('ik,jl,klm->ijm', b_u_tensor, db_v_tensor, control_points_tensor)
    second_derivative_uu = torch.einsum('ik,jl,klm->ijm', ddb_u_tensor, b_v_tensor, control_points_tensor)
    second_derivative_vv = torch.einsum('ik,jl,klm->ijm', b_u_tensor, ddb_v_tensor, control_points_tensor)
    second_derivative_uv = torch.einsum('ik,jl,klm->ijm', db_u_tensor, db_v_tensor, control_points_tensor)

    normal_vectors = torch.cross(derivative_u, derivative_v, dim=2)
    normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=2, keepdim=True)

    E = torch.sum(derivative_u * derivative_u, dim=2)
    F = torch.sum(derivative_u * derivative_v, dim=2)
    G = torch.sum(derivative_v * derivative_v, dim=2)

    L = torch.sum(normal_vectors * second_derivative_uu, dim=2)
    M = torch.sum(normal_vectors * second_derivative_uv, dim=2)
    N = torch.sum(normal_vectors * second_derivative_vv, dim=2)

    curvature_gaussian = (L * N - M * M) / (E * G - F * F)
    curvature_mean = (E * N + G * L - 2 * F * M) / (2 * (E * G - F * F))

    return torch.nan_to_num(curvature_gaussian, nan=.0), torch.nan_to_num(curvature_mean, nan=.0)

def compute_spline_curvature_batch(control_points_batch, degrees_u, degrees_v, knots_u_batch, knots_v_batch,
                                   num_points=100):
    def b_spline_basis_function(i, k, t, knots):
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
        else:
            coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
            coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[
                i + 1] else 0
            return coef1 * b_spline_basis_function(i, k - 1, t, knots) + coef2 * b_spline_basis_function(i + 1, k - 1,
                                                                                                         t, knots)

    def b_spline_basis_function_derivative(i, k, t, knots):
        if k == 0:
            return 0.0
        else:
            coef1 = k / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
            coef2 = k / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
            return coef1 * b_spline_basis_function(i, k - 1, t, knots) - coef2 * b_spline_basis_function(i + 1, k - 1,
                                                                                                         t, knots)

    num_patches = control_points_batch.shape[0]
    num_points_u = num_points
    num_points_v = num_points

    # Compute basis functions and their derivatives for all patches
    u_values = torch.linspace(knots_u_batch[0][degrees_u], knots_u_batch[0][-degrees_u - 1], num_points_u)
    v_values = torch.linspace(knots_v_batch[0][degrees_v], knots_v_batch[0][-degrees_v - 1], num_points_v)

    b_u_tensors = torch.stack([torch.tensor(
        [[b_spline_basis_function(k, degrees_u, u, knots_u_batch[p]) for k in range(control_points_batch.shape[1])] for
         u in u_values], dtype=torch.float32) for p in range(num_patches)])
    b_v_tensors = torch.stack([torch.tensor(
        [[b_spline_basis_function(l, degrees_v, v, knots_v_batch[p]) for l in range(control_points_batch.shape[2])] for
         v in v_values], dtype=torch.float32) for p in range(num_patches)])

    db_u_tensors = torch.stack([torch.tensor([[b_spline_basis_function_derivative(k, degrees_u, u, knots_u_batch[p]) for
                                               k in range(control_points_batch.shape[1])] for u in u_values],
                                             dtype=torch.float32) for p in range(num_patches)])
    db_v_tensors = torch.stack([torch.tensor([[b_spline_basis_function_derivative(l, degrees_v, v, knots_v_batch[p]) for
                                               l in range(control_points_batch.shape[2])] for v in v_values],
                                             dtype=torch.float32) for p in range(num_patches)])

    ddb_u_tensors = torch.stack([torch.tensor([[b_spline_basis_function_derivative(k, degrees_u - 1, u,
                                                                                   knots_u_batch[p]) for k in
                                                range(control_points_batch.shape[1])] for u in u_values],
                                              dtype=torch.float32) for p in range(num_patches)])
    ddb_v_tensors = torch.stack([torch.tensor([[b_spline_basis_function_derivative(l, degrees_v - 1, v,
                                                                                   knots_v_batch[p]) for l in
                                                range(control_points_batch.shape[2])] for v in v_values],
                                              dtype=torch.float32) for p in range(num_patches)])

    # Compute surface points and their derivatives using einsum
    surface_points = torch.einsum('bik,bjl,bklm->bijm', b_u_tensors, b_v_tensors, control_points_batch)
    derivative_u = torch.einsum('bik,bjl,bklm->bijm', db_u_tensors, b_v_tensors, control_points_batch)
    derivative_v = torch.einsum('bik,bjl,bklm->bijm', b_u_tensors, db_v_tensors, control_points_batch)
    second_derivative_uu = torch.einsum('bik,bjl,bklm->bijm', ddb_u_tensors, b_v_tensors, control_points_batch)
    second_derivative_vv = torch.einsum('bik,bjl,bklm->bijm', b_u_tensors, ddb_v_tensors, control_points_batch)
    second_derivative_uv = torch.einsum('bik,bjl,bklm->bijm', db_u_tensors, db_v_tensors, control_points_batch)

    # Compute normal vectors
    normal_vectors = torch.cross(derivative_u, derivative_v, dim=3)
    normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=3, keepdim=True)

    # Compute the first fundamental form
    E = torch.sum(derivative_u * derivative_u, dim=3)
    F = torch.sum(derivative_u * derivative_v, dim=3)
    G = torch.sum(derivative_v * derivative_v, dim=3)

    # Compute the second fundamental form
    L = torch.sum(normal_vectors * second_derivative_uu, dim=3)
    M = torch.sum(normal_vectors * second_derivative_uv, dim=3)
    N = torch.sum(normal_vectors * second_derivative_vv, dim=3)

    # Compute Gaussian and mean curvature
    curvature_gaussian = (L * N - M * M) / (E * G - F * F)
    curvature_mean = (E * N + G * L - 2 * F * M) / (2 * (E * G - F * F))

    return torch.nan_to_num(curvature_gaussian, nan=.0),  torch.nan_to_num(curvature_mean, nan=.0)


# Test function to compare both implementations
def test_compute_spline_curvature():
    # Example input data
    control_points_batch = torch.tensor([
        # Control points for the first patch
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ],
        # Control points for the second patch
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 1.0], [2.0, 2.0, 0.0]],
        ]
    ], dtype=torch.float32)

    control_points_batch = torch.randn_like(control_points_batch)

    degrees_u = 2
    degrees_v = 2
    knots_u_batch = [torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32),
                     torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)]
    knots_v_batch = [torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32),
                     torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)]

    curvature_gaussian_batch, curvature_mean_batch = compute_spline_curvature_batch(control_points_batch, degrees_u,
                                                                                    degrees_v, knots_u_batch,
                                                                                    knots_v_batch)

    curvature_gaussian_iter = []
    curvature_mean_iter = []
    for i in range(control_points_batch.shape[0]):
        c_points = control_points_batch[i].numpy()
        k_u = knots_u_batch[i].numpy()
        k_v = knots_v_batch[i].numpy()
        c_g, c_m = compute_spline_curvature_iterative(c_points, degrees_u, degrees_v, k_u, k_v)
        curvature_gaussian_iter.append(c_g)
        curvature_mean_iter.append(c_m)

    curvature_gaussian_iter = torch.stack(curvature_gaussian_iter)
    curvature_mean_iter = torch.stack(curvature_mean_iter)

    assert torch.allclose(curvature_gaussian_batch, curvature_gaussian_iter, atol=1e-6), "Gaussian curvature mismatch"
    assert torch.allclose(curvature_mean_batch, curvature_mean_iter, atol=1e-6), "Mean curvature mismatch"

    print("Test passed: Both implementations produce the same results.")


# Run the test
test_compute_spline_curvature()
