import torch
from scipy.special import comb


# def bernstein_poly(i, n, t):
#     return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
#
#
# def bezier_surface_point(u, v, udim, vdim):
#     point = torch.zeros(3)
#     for i in range(udim):
#         for j in range(vdim):
#             bernstein_u = bernstein_poly(i, udim - 1, u)
#             bernstein_v = bernstein_poly(j, vdim - 1, v)
#             point += control_points[i, j] * bernstein_u * bernstein_v
#     return point


class BezierSurface:
    def __init__(self, control_points, degrees_u=50, degrees_v=50, device='cpu'):
        self._control_points = control_points
        self._degrees_u = degrees_u
        self._degrees_v = degrees_v
        self._surface = self.build_bezier_surface(self._control_points, self._degrees_u, self._degrees_v)

    def build_bezier_surface(self, control_points, u_steps=50, v_steps=50):
        """
        Create a Bézier surface from a tensor of control points.

        Parameters:
        - control_points: a tensor of shape (m, n, 3) representing the control points in 3D space
        - u_steps: number of steps to sample in the u direction
        - v_steps: number of steps to sample in the v direction

        Returns:
        - A tensor of shape (u_steps, v_steps, 3) representing the Bézier surface points
        """
        udim, vdim, _ = control_points.shape

        u = torch.linspace(0, 1, u_steps)
        v = torch.linspace(0, 1, v_steps)
        surface_points = torch.zeros(u_steps, v_steps, 3)

        for i in range(u_steps):
            for j in range(v_steps):
                surface_points[i, j] = self.bezier_surface_point(u[i], v[j], udim, vdim)

        return surface_points

    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bezier_surface_point(self, u, v, udim, vdim):
        point = torch.zeros(3)
        for i in range(udim):
            for j in range(vdim):
                bernstein_u = self.bernstein_poly(i, udim - 1, u)
                bernstein_v = self.bernstein_poly(j, vdim - 1, v)
                point += self._control_points[i, j] * bernstein_u * bernstein_v
        return point
    def get_surface_points(self):
        return self._surface

    def debug(self):


        # Plotting the surface
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = self._surface[:, :, 0].numpy(), self._surface[:, :, 1].numpy(), self._surface[:, :, 2].numpy()
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        # Plot control points for reference
        cp_x = self._control_points[:, :, 0].numpy()
        cp_y = self._control_points[:, :, 1].numpy()
        cp_z = self._control_points[:, :, 2].numpy()
        ax.scatter(cp_x, cp_y, cp_z, color='red')
        for i in range(cp_x.shape[0]):
            for j in range(cp_x.shape[1]):
                ax.scatter(cp_x[i, j], cp_y[i, j], cp_z[i, j])



        plt.show()
#
# # Example usage
# control_points = torch.tensor([
#     [[0, 0, 0], [1, 0, -20], [2, 0, -10]],
#     [[0, 1, 1], [1, 1, -20], [2, 1, 10]],
#     [[0, 2, 0], [1, 2, -20], [2, 2, -10]]
# ], dtype=torch.float32)
#
# model = BezierSurface(control_points)
# # surface = model.build_bezier_surface(control_points, u_steps=50, v_steps=50)
# model.debug()

