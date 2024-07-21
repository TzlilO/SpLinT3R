from model import *
from utils.data import load_teapot
import open3d as o3d
from scipy.interpolate import LSQSphereBivariateSpline


def mesh_to_spline_surface(mesh_file):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # Sample points from the mesh
    point_cloud = mesh.sample_points_uniformly(number_of_points=1000)
    points = np.asarray(point_cloud.points)

    # Parameterize points (for simplicity, using first two dimensions as parameters)
    u = points[:, 0]
    v = points[:, 1]

    # Fit a B-spline surface
    degree_u = 3
    degree_v = 3
    knots_u = np.linspace(min(u), max(u), degree_u + 1)
    knots_v = np.linspace(min(v), max(v), degree_v + 1)

    # Use least-squares Bivariate Spline
    spline_surface = LSQSphereBivariateSpline(u, v, points[:, 2], knots_u, knots_v, kx=degree_u, ky=degree_v)

    return spline_surface