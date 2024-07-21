import numpy as np
import matplotlib.pyplot as plt
def plot_b_spline_surface_with_derivatives(surface_points, derivative_points_u, derivative_points_v,
                                            control_points, curvature_gaussians, sample_points=10):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface_points = surface_points.clone().detach().cpu().numpy()
    curvature_gaussians = curvature_gaussians.clone().detach().cpu().numpy()
    control_points = control_points.clone().detach().cpu().numpy()
    derivative_points_u = derivative_points_u.clone().detach().cpu().numpy()
    derivative_points_v = derivative_points_v.clone().detach().cpu().numpy()
    draw_surface(ax, surface_points)

    draw_curvatures(ax, curvature_gaussians, surface_points)

    draw_control_points(ax, control_points)

    draw_uv_tangents(ax, derivative_points_u, derivative_points_v, sample_points, surface_points)
    plt.show()

def hist_1D(density_distribution):
    mn = density_distribution.cpu().numpy().min()
    mx = density_distribution.cpu().numpy().max()
    hist, bins = np.histogram(density_distribution.cpu().numpy(), bins=1000, range=(mn, mx))
    # Plot histogram
    plt.hist(bins[:-1], bins, weights=hist)
    plt.xlabel('Density Distribution')
    plt.ylabel('Frequency')
    plt.title('Histogram of Density Distribution')
    plt.show()
def draw_curvatures(ax, curvature_gaussians, surface_points):
    curvatures = curvature_gaussians  # or curvature_means depending on what you want to visualize
    curvatures = np.abs(curvatures)
    ax.plot_surface(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], rstride=1, cstride=1,
                    facecolors=plt.cm.hot(curvatures), alpha=1)

def draw_surface(ax, surface_points):
    # curvatures = curvature_gaussians  # or curvature_means depending on what you want to visualize
    # curvatures = torch.abs(curvatures)
    ax.plot_surface(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2][None], rstride=1, cstride=1)
def draw_control_points(ax, control_points):
    control_points = control_points.clone().detach().cpu()
    # Plot control points
    for i in range(len(control_points)):
        ax.plot(control_points[i, 0], control_points[i, 1], control_points[i, 2], 'ro-')
    for j in range(len(control_points[0])):
        ax.plot(control_points[j, 0], control_points[j, 1], control_points[j, 2], 'ro-')


def draw_uv_tangents(ax, derivative_points_u, derivative_points_v, sample_points, surface_points):
    # Plot derivatives w.r.t du and dv
    # num_points_u, num_points_v = surface_points.shape[0]
    num_points = surface_points.shape[0]
    sampled_indices = np.linspace(0, num_points - 1, sample_points, dtype=int)
    # sampled_indices_v = np.linspace(0, num_points_v - 1, sample_points, dtype=int)
    # sampled_indices_u = np.linspace(0, num_points_u - 1, sample_points, dtype=int)
    for i in sampled_indices:
        # for j in sampled_indices_v:
        mag_u = np.linalg.norm(derivative_points_u[i])
        mag_v = np.linalg.norm(derivative_points_v[i])
        ax.quiver(surface_points[i, 0], surface_points[i, 1], surface_points[i, 2],
                  derivative_points_u[i, 0], derivative_points_u[i,1], derivative_points_u[i, 2],
                  color='r', length=mag_u * 0.1, normalize=True)
                  # ,label='u-derivative' if i == sampled_indices_u[0] and j == sampled_indices[0] else "")
        ax.quiver(surface_points[i, 0], surface_points[i, 1], surface_points[i, 2],
                  derivative_points_v[i, 0], derivative_points_v[i, 1], derivative_points_v[i, 2],
                  color='b', length=mag_v * 0.1, normalize=True)
                  # ,
                  # label='v-derivative' if i == sampled_indices[0] and j == sampled_indices[0] else "")

