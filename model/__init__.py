import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyElement, PlyData
from torch.nn import functional as F
from .spline_utils import to_quaternions, b_spline_basis_functions_and_derivatives, patch_subdivision, \
    normalize_point_cloud, compute_scale_factors, \
    DecayingCosineAnnealingWarmRestarts, inverse_sigmoid, Gaussians, plot_lr_schedule, quaternion_from_two_vectors, \
    analyze_gradient_trend, analyze_gradient_trend_per_patch, patch_subdivisions, SH_interpolation
import os

RESCALING_UP = 2.
RESCALING_DOWN = .9


class SplineModel(nn.Module):
    def __init__(self, patches, device='cuda', resolution=16, debug=False, splitting_interval_every=1000, step_size=1):
        super(SplineModel, self).__init__()
        self.step_size = step_size
        self.splitting_interval_every = splitting_interval_every
        self.device = device
        num_patches = len(patches)
        self.num_patches = num_patches
        self.__debug = debug
        self._res = resolution
        self._max_sh_degree = 4
        self.active_sh_degree = 0
        ctrl_pts = normalize_point_cloud(torch.stack([patch[2] for patch in patches]))

        noise = torch.randn_like(ctrl_pts, requires_grad=True) * 0.
        self.control_points = nn.Parameter(ctrl_pts + noise, requires_grad=True)

        sph_harm_features = torch.ones(
            (self.num_patches, ctrl_pts.shape[1], ctrl_pts.shape[2], (self._max_sh_degree + 1) ** 2, 3), device=device).contiguous()
        self.spherical_harmonics_dc = nn.Parameter(sph_harm_features[:, :, :, :1, :], requires_grad=True)
        self.spherical_harmonics_rest = nn.Parameter(sph_harm_features[:, :, :, 1:, :]*.0, requires_grad=True)
        self.scale_params = nn.Parameter(torch.ones((num_patches, 2), device=device), requires_grad=True)
        self.quaternion_params = nn.Parameter(torch.ones((num_patches, 4), device=device), requires_grad=True)

        # learning rates
        self.position_lr = 1e-4
        self.nurbs_weights_lr = 1e-6
        self.feature_lr = 5e-2
        self.scale_params_lr = 1e-2
        self.quaternion_params_lr = 5e-2
        lr_final_factor = 0.1
        self.position_lr_final = self.position_lr * lr_final_factor
        self.feature_lr_final = self.feature_lr * lr_final_factor
        self.scale_params_lr_final = self.scale_params_lr * lr_final_factor
        self.quaternion_params_lr_final = self.quaternion_params_lr * lr_final_factor
        self.nurbs_weights_lr_final = self.nurbs_weights_lr * lr_final_factor

        # Variables to store computation results
        self._patches = patches
        self._degrees_u = torch.empty(0)
        self._degrees_v = torch.empty(0)
        self._knots_u = torch.empty(0)
        self._knots_v = torch.empty(0)
        self.surface_points = torch.empty(0)
        self.surface_normals = torch.empty(0)

        # For 3D Gaussians post-processing sampling
        self.gaussians = None
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.__opacity = torch.ones((num_patches * resolution * resolution, 1), requires_grad=True, device=self.device)

        self.nurbs_weights = nn.Parameter(torch.ones((self.num_patches, self.control_points.shape[1], self.control_points.shape[2]),
                                                     dtype=torch.float32, device=self.device, requires_grad=True))
        self.set_surface_data()
        self.training_setup()

    def set_surface_data(self):
        # TODO: Check if that's correct and if so, reduce to 1-tensor 'fits them all' instead of (num_patches * tensors)

        self._degrees_u = torch.tensor([patch[0] for patch in self._patches], dtype=torch.int32, device=self.device)
        self._degrees_v = torch.tensor([patch[1] for patch in self._patches], dtype=torch.int32, device=self.device)
        self._knots_u = torch.tensor([[0] * (patch[0] + 1) + [1] * (patch[0] + 1) for patch in self._patches],
                                     dtype=torch.float32, device=self.device)
        self._knots_v = torch.tensor([[0] * (patch[1] + 1) + [1] * (patch[1] + 1) for patch in self._patches],
                                     dtype=torch.float32, device=self.device)
        deg_u = self._degrees_u[0]+1
        deg_v = self._degrees_v[0]+1

        # TODO: Squeeze the batch dimension and modify the gradients_handler accordingly
        self.b_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.b_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()
        self.db_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.db_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()
        self.ddb_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.ddb_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()
        self.dus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.ddus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.ddvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dduvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32,requires_grad=True)
        self.areas = torch.empty((self.num_patches, 1), device=self.device, dtype=torch.float32,requires_grad=True)
        self.control_points_grads = torch.zeros((self.splitting_interval_every//self.step_size, self.control_points.shape[0], self.control_points.shape[1], self.control_points.shape[2], self.control_points.shape[3]), device=self.device)
        self.spherical_harmonics_dc_grads = torch.zeros((self.splitting_interval_every//self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], 1, 3), device=self.device)
        self.spherical_harmonics_rest_grads = torch.zeros((self.splitting_interval_every//self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2 - 1, 3), device=self.device)
        self.__opacity = torch.ones((self.num_patches * self._res * self._res, 1), requires_grad=True,
                                    device=self.device)
        for patch_id in range(self.num_patches):
            self.preprocess_splines(patch_id)

    def oneupSHdegree(self):
        if self.active_sh_degree < self._max_sh_degree:
            self.active_sh_degree += 1

    def preprocess_splines(self, patch_id):
        device = self.device
        num_points_u = num_points_v = self._res
        degree_u = self._degrees_u[patch_id]
        degree_v = self._degrees_v[patch_id]
        knots_u = self._knots_u[patch_id] #+ patch_id*2
        knots_v = self._knots_v[patch_id] #+ patch_id*2 + 1
        u_values = torch.linspace(knots_u[degree_u], knots_u[-degree_u - 1], num_points_u)
        v_values = torch.linspace(knots_v[degree_v], knots_v[-degree_v - 1], num_points_v)

        self.b_u_tensor[patch_id], self.db_u_tensor[patch_id],  self.ddb_u_tensor[patch_id] = b_spline_basis_functions_and_derivatives(degree_u, knots_u.cpu(), u_values, device)
        self.b_v_tensor[patch_id], self.db_v_tensor[patch_id],  self.ddb_v_tensor[patch_id] = b_spline_basis_functions_and_derivatives(degree_v, knots_v.cpu(), v_values, device)

        self.U_tensors = torch.stack([self.b_u_tensor, self.db_u_tensor, self.ddb_u_tensor])
        self.V_tensors = torch.stack([self.b_v_tensor, self.db_v_tensor, self.ddb_v_tensor])
        self.sh_features = torch.concatenate([self.spherical_harmonics_dc, self.spherical_harmonics_rest], dim=-2)


    def fit_nurbs(self):
        epsilon = 100  # Small value to prevent division by zero
        denominator = torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.b_v_tensor, self.nurbs_weights).flatten(start_dim=0).unsqueeze(-1)
        denominator[denominator == 0] = 1
        # denominator
        weighted_control_points = self.control_points * self.nurbs_weights.unsqueeze(-1)


        self.xyz = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, weighted_control_points).view(-1, 3) / denominator


        self.dus = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.b_v_tensor,
                                            weighted_control_points).reshape(-1, 3)

        self.dvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.db_v_tensor,
                                            weighted_control_points).reshape(-1, 3)

        self.ddus = torch.einsum('aik,ajl,aklm->aijm', self.ddb_u_tensor, self.b_v_tensor,
                           weighted_control_points).reshape(-1, 3)

        self.ddvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.ddb_v_tensor,
                            weighted_control_points).reshape(-1, 3)


        self.dduvs = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.db_v_tensor,
                            weighted_control_points).reshape(-1, 3)

        # SH sampling
        sh_features = torch.concatenate([self.spherical_harmonics_dc, self.spherical_harmonics_rest], dim=-2)
        SHs_sampling = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, sh_features.view(self.num_patches, 4, 4, -1))

        ################################
        ########### Scaling ###########
        ################################

        # Compute patch areas and bounding box volumes
        patch_areas = self.compute_patches_area()

        # Compute scaling factors based on patch area, bounding box volume, and curvature
        scale_params = torch.cat([self.scale_params, torch.ones((self.scale_params.shape[0], 1), dtype=torch.float32, device=self.device, requires_grad=True) * 1e-5], dim=-1)
        scale_factors = patch_areas.unsqueeze(-1)
        scale_factors = ((scale_factors / (self._res ** 2)) * scale_params).repeat_interleave(self._res ** 2, dim=0)

        # Compute the normal vector (n = r_u x r_v)
        self.surface_normals = F.normalize(torch.cross(self.dus, self.dvs, dim=-1),  p=2, dim=-1)

        # Compute the first fundamental form coefficients
        E = (self.dus * self.dus).sum(-1)
        C = (self.dus * self.dvs).sum(-1)
        G = (self.dvs * self.dvs).sum(-1)

        # Compute the second fundamental form coefficients
        L = (self.surface_normals * self.ddus).sum(-1)
        M = (self.surface_normals * self.ddvs).sum(-1)
        N = (self.surface_normals * self.dduvs).sum(-1)

        # Compute mean curvature
        H = (E * N + G * L - 2 * C * M) / (2 * (E * G - C ** 2) + epsilon)

        # Compute Gaussian curvature
        K = (L * N - M ** 2) / (E * G - C ** 2 + epsilon)

        # Compute principal curvatures
        k1 = H + torch.sqrt(H ** 2 - K + epsilon)
        k2 = H - torch.sqrt(H ** 2 - K + epsilon)

        is_locally_elliptic = (k1 * k2 > 0)
        is_locally_hyperbolic = (k1 * k2 < 0)
        is_locally_parabolic = ((k1 == 0) | (k2 == 0))

        # scaling = self.scale_mapping(epsilon, is_locally_elliptic, is_locally_hyperbolic, is_locally_parabolic, k1, k2,
        #                              scale_factors)
        scaling = torch.zeros_like(k1).unsqueeze(-1).repeat(1, 3)
        scaling[is_locally_elliptic] = torch.exp(torch.stack(
            [k1[is_locally_elliptic].abs(), k2[is_locally_elliptic].abs(), (k1[is_locally_elliptic] + k2[is_locally_elliptic]).abs() / 2],
            dim=-1))
        scaling[is_locally_hyperbolic] = torch.exp(torch.stack([k1[is_locally_hyperbolic].abs(), k2[is_locally_hyperbolic].abs(),
                                                                (k1[is_locally_hyperbolic] - k2[
                                                                    is_locally_hyperbolic]).abs() / 2],
                                                               dim=-1))
        scaling[is_locally_parabolic] = torch.exp(
            torch.stack(
                [k1[is_locally_parabolic].abs(), k2[is_locally_parabolic].abs(), torch.zeros_like(k1[is_locally_parabolic])],
                dim=-1))
        scaling = scaling / scaling.norm(p=1, dim=-1, keepdim=True)
        scaling = (scale_factors.abs().sqrt().log() - (scaling + epsilon).log())
        scaling.clamp_(max=-1, min=-9)
        # Compute rotation
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.xyz.shape[0], 3)
        rotation = quaternion_from_two_vectors(up, self.surface_normals.reshape(-1, 3)) * self.quaternion_params.repeat_interleave(self._res ** 2, dim=0)
        # Reshape surface normals
        self.surface_normals = self.surface_normals.view(self.num_patches, self._res, self._res, 3)

        return self.xyz, self.__opacity, scaling, rotation, SHs_sampling.view(-1, (self._max_sh_degree + 1) ** 2, 3)


    def scale_mapping(self, epsilon, is_locally_elliptic, is_locally_hyperbolic, is_locally_parabolic, k1, k2,
                      scale_factors):

        scaling = torch.zeros_like(k1).unsqueeze(-1).repeat(1, 3)
        scaling[is_locally_elliptic] = torch.exp(torch.stack(
            [k1[is_locally_elliptic].abs(), k2[is_locally_elliptic].abs(), (k1[is_locally_elliptic] + k2[is_locally_elliptic]).abs() / 2],
            dim=-1))
        scaling[is_locally_hyperbolic] = torch.exp(torch.stack([k1[is_locally_hyperbolic].abs(), k2[is_locally_hyperbolic].abs(),
                                                                (k1[is_locally_hyperbolic] - k2[
                                                                    is_locally_hyperbolic]).abs() / 2],
                                                               dim=-1))
        scaling[is_locally_parabolic] = torch.exp(
            torch.stack(
                [k1[is_locally_parabolic].abs(), k2[is_locally_parabolic].abs(), torch.zeros_like(k1[is_locally_parabolic])],
                dim=-1))
        scaling = scaling / scaling.norm(p=1, dim=-1, keepdim=True)
        scaling = (scale_factors.abs().sqrt().log() - (scaling + epsilon).log())
        scaling.clamp_(max=-1, min=-9)
        return scaling


    ########################################################################
    ############################# DENSIFICATION ############################
    ########################################################################

    def visibility_filter_to_patches(self, visibility_filter: torch.Tensor):
        mask = visibility_filter.reshape(self.num_patches, -1).sum(dim=-1)
        mask[mask < self._res**2 / 2] = 0
        return mask.type(torch.BoolTensor)


    def grads_handler(self, iteration, visibility_filter, grad_clipper=.0, retain_grad=False):
        mask = self.visibility_filter_to_patches(visibility_filter)
        # Compute the gradient of self.control_points
        pos = (iteration//self.step_size - 1 )% (self.splitting_interval_every // self.step_size)
        self.control_points_grads[pos][mask] = torch.norm(self.control_points.grad[mask], dim=-1, keepdim=True) / self.splitting_interval_every
        # Compute the gradient of self.spherical_harmonics_dc
        self.spherical_harmonics_dc_grads[pos][mask] = torch.norm(self.spherical_harmonics_dc.grad[mask], dim=-1, keepdim=True)  / self.splitting_interval_every
        # Compute the gradient of self.spherical_harmonics_rest
        self.spherical_harmonics_rest_grads[pos][mask] = torch.norm(self.spherical_harmonics_rest.grad[mask], dim=-1, keepdim=True)  / self.splitting_interval_every

        if grad_clipper:
            torch.nn.utils.clip_grad_norm_(self.control_points, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.spherical_harmonics_dc, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.spherical_harmonics_rest, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.scale_params, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.quaternion_params, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.nurbs_weights, max_norm=grad_clipper)

        if retain_grad:
            self.control_points.retain_grad()
            self.spherical_harmonics_dc.retain_grad()
            self.spherical_harmonics_rest.retain_grad()
            self.scale_params.retain_grad()
            self.quaternion_params.retain_grad()
            self.nurbs_weights.retain_grad()


    def patches_to_split(self):
        # Shape is (BATCH x Patches x 4 x 4 x 3) representing a batch of patches of a 3D surface
        control_points_grads = self.control_points_grads

        # Shape is (BATCH x Patches x 4 x 4 x Max_shperical_harmonic_coefficients x 3) representing a SH features corresponds to each control point
        spherical_harmonics_dc_grads = self.spherical_harmonics_dc_grads.reshape(self.splitting_interval_every//self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)
        spherical_harmonics_rest_grads = self.spherical_harmonics_rest_grads.reshape(self.splitting_interval_every//self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)

        control_points_converging = analyze_gradient_trend_per_patch(control_points_grads, top_k=self.num_patches //2)
        SH_dc_converging = analyze_gradient_trend_per_patch(spherical_harmonics_dc_grads, top_k=self.num_patches // 2)
        SH_rest_converging = analyze_gradient_trend_per_patch(spherical_harmonics_rest_grads, top_k=self.num_patches // 2)
        largest_patches_areas, largest_patches_indices = torch.topk(self.areas.flatten(), k=self.num_patches // 2)
        most_largest_patches = torch.ones_like(control_points_converging['top_k_mask'])
        most_largest_patches[largest_patches_indices] = True
        return (control_points_converging['top_k_mask'] & SH_dc_converging['top_k_mask'] & SH_rest_converging['top_k_mask'] & most_largest_patches)

    def patch_upsampler(self):
        with torch.no_grad():
            print(f"\nnum patches before: {self.num_patches}")
            patch_to_split_indices = self.patches_to_split()
            self.control_points_upsampling(patch_to_split_indices)
            print(f"num patches after: {self.num_patches}\n")

    def noise_scales(self):
        with torch.no_grad():
            self.scale_params.data = self.scale_params.data * RESCALING_DOWN
    def control_points_upsampling(self, do_upsampling_mask):
        non_zeros = do_upsampling_mask.sum().item()
        features = torch.cat([self.spherical_harmonics_dc.clone().detach(), self.spherical_harmonics_rest.clone().detach()], dim=-2).reshape(self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2, 3)
        features_to_optimize = features[do_upsampling_mask]

        control_points_to_optimize = self.control_points[do_upsampling_mask].clone().detach()
        features_other = features[~do_upsampling_mask]
        old_control_points = self.control_points[~do_upsampling_mask]

        # Prepare updated parameters values
        new_sh =SH_interpolation(features_to_optimize)
        splitted_control_points = patch_subdivisions(control_points_to_optimize)
        new_control_points = torch.cat((old_control_points, splitted_control_points)).contiguous()
        new_nurbs_weights = torch.cat((self.nurbs_weights[~do_upsampling_mask], self.nurbs_weights[do_upsampling_mask].repeat_interleave(4, dim=0))).contiguous()
        sph_harm_features = torch.cat((features_other, new_sh))
        new_features_dc = sph_harm_features[:,:,:, :1, :].contiguous()
        new_features_rest = sph_harm_features[:,:,:, 1:, :].contiguous()

        if non_zeros > 0:
            existing_scaling = self.scale_params[~do_upsampling_mask]
            splitted_scaling = self.scale_params[do_upsampling_mask].repeat_interleave(4, dim=0) * RESCALING_UP
            new_scaling = torch.cat([existing_scaling, splitted_scaling], dim=0).contiguous() * 0.8
            new_rotation = torch.cat([self.quaternion_params[~do_upsampling_mask], self.quaternion_params[do_upsampling_mask].repeat_interleave(4, dim=0)], dim=0).contiguous()
        else:
            new_scaling = self.scale_params[~do_upsampling_mask]
            new_rotation = self.quaternion_params[~do_upsampling_mask]

        # From Gaussian Splatting
        d = {"control_points": new_control_points,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "scale_params": new_scaling,
             "quaternion_params": new_rotation,
             "nurbs_weights": new_nurbs_weights
             }

        optimizable_tensors = self.override_tensors_in_optimizer(d)
        self.control_points = optimizable_tensors["control_points"]
        self.spherical_harmonics_dc = optimizable_tensors["f_dc"]
        self.spherical_harmonics_rest = optimizable_tensors["f_rest"]
        self.scale_params = optimizable_tensors["scale_params"]
        self.quaternion_params = optimizable_tensors["quaternion_params"]
        self.nurbs_weights = optimizable_tensors["nurbs_weights"]
        self.num_patches = len(self.control_points)
        self._patches = [(self._degrees_u[0], self._degrees_v[0], self.control_points[patchId]) for patchId in range(self.num_patches)]
        self.splitting_interval_every *= 2
        self.set_surface_data()
        self.param_count_report()

    def inverse_sigmoid(self, x):
        return torch.log(x / (1 - x))

    def sample_gaussians(self):
        xyz, opacity, scaling, rotation, features = self.fit_nurbs()
        self.gaussians = Gaussians(xyz=xyz, opacity=self.inverse_sigmoid(opacity), scaling=scaling, rotation=rotation,
                                   features=features)

    ########################################################################
    ##################### OPTIMIZATION CONFIGURATION #######################
    ########################################################################

    def training_setup(self):
        self.param_groups = [
            {'params': self.control_points, 'lr': self.position_lr, 'min_lr': self.position_lr_final, "name": "control_points"},
            {'params': self.spherical_harmonics_dc, 'lr': self.feature_lr, 'min_lr': self.feature_lr_final,"name": "f_dc"},
            {'params': self.spherical_harmonics_rest, 'lr': self.feature_lr / 10, 'min_lr': self.feature_lr_final / 10, "name": "f_rest"},
            {'params': self.scale_params,  'lr': self.scale_params_lr, 'min_lr': self.scale_params_lr_final,"name": "scale_params"},
            {'params': self.quaternion_params, 'lr': self.quaternion_params_lr, 'min_lr': self.quaternion_params_lr_final, "name": "quaternion_params"},
            {'params': self.nurbs_weights, 'lr': self.nurbs_weights_lr, 'min_lr': self.nurbs_weights_lr_final, "name": "nurbs_weights"}
        ]

        self.optimizers = []
        self.schedulers = []

        for group in self.param_groups:
            if group['name'] == 'scale_params':
                alpha = .4
                beta = .1
            else:
                alpha = 0.4
                beta = 0.1
            optimizer = torch.optim.Adam([group['params']], lr=group['lr'], eps=1e-10)
            scheduler = DecayingCosineAnnealingWarmRestarts(optimizer, T_0=self.splitting_interval_every//self.step_size, T_mult=2, eta_min=group['min_lr'], alpha=alpha, beta=beta, verbose=False)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        if self.device:
            plot_lr_schedule(self.schedulers[0], num_epochs=self.splitting_interval_every // self.step_size * 10)


    def step(self, iteration, visibility_filter):
        self.grads_handler(iteration, visibility_filter, retain_grad=True, grad_clipper=0)
        # Update optimizers
        for scheduler in self.schedulers:
            scheduler.step()
            scheduler.optimizer.step()

        # Clear gradients after optimization
        for scheduler in self.schedulers:
            scheduler.optimizer.zero_grad(set_to_none=True)

    def override_tensors_in_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for idx, opt in enumerate(self.optimizers):
            try:
                group = opt.param_groups[0]
                assert len(group["params"]) == 1
                new_tensor = tensors_dict[
                    self.param_groups[idx]["name"]]  # This is the new tensor to replace the existing one
                stored_state = self.optimizers[idx].state.get(group['params'][0], None)
                if stored_state is not None:
                    # Resetting optimizer states (assuming momentum buffers etc. should be reinitialized)
                    stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)

                    # Remove old state references
                    del self.optimizers[idx].state[group['params'][0]]

                    # Replace the parameter tensor
                    group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                    self.optimizers[idx].state[group['params'][0]] = stored_state
                else:
                    # Simply replace the tensor if there was no stored state
                    group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))

                optimizable_tensors[self.param_groups[idx]["name"]] = group["params"][0]
            except KeyError:
                continue
        torch.cuda.empty_cache()

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    ########################################################################
    ########################## UTILITIES ###################################
    ########################################################################

    def construct_list_of_attributes(self, features_dc, features_rest, scaling, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))

        for i in range(features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def compute_patches_area(self):
        """
        Computes the area of the NURBS surface patch.

        Parameters:
            xyz (torch.Tensor): Surface points (shape: BATCH_SIZE x res x res x 3).
            dus (torch.Tensor): First derivatives with respect to u (shape: BATCH_SIZE x res x res x 3).
            dvs (torch.Tensor): First derivatives with respect to v (shape: BATCH_SIZE x res x res x 3).

        Returns:
            torch.Tensor: Area of the surface patch for each element in the batch (shape: BATCH_SIZE).
        """
        # Compute the cross product of the derivatives to get the differential area elements
        cross_product = torch.cross(self.dus.reshape(self.num_patches, self._res,self. _res, 3), self.dvs.reshape(self.num_patches, self._res,self. _res, 3), dim=-1)
        differential_area = torch.norm(cross_product, dim=-1)

        # Integrate the differential area element over the parameter domain using the trapezoidal rule
        self.areas = torch.trapz(torch.trapz(differential_area, dim=2), dim=1).unsqueeze(1).unsqueeze(2)
        return self.areas.squeeze(2).squeeze(1)

    def compute_patches_area_explicit(self, dus, dvs):
        """
        Computes the area of the NURBS surface patch.

        Parameters:
            xyz (torch.Tensor): Surface points (shape: BATCH_SIZE x res x res x 3).
            dus (torch.Tensor): First derivatives with respect to u (shape: BATCH_SIZE x res x res x 3).
            dvs (torch.Tensor): First derivatives with respect to v (shape: BATCH_SIZE x res x res x 3).

        Returns:
            torch.Tensor: Area of the surface patch for each element in the batch (shape: BATCH_SIZE).
        """
        # Compute the cross product of the derivatives to get the differential area elements
        cross_product = torch.cross(dus, dvs, dim=-1)
        differential_area = torch.norm(cross_product, dim=-1)

        # Integrate the differential area element over the parameter domain using the trapezoidal rule
        return torch.trapz(torch.trapz(differential_area, dim=2), dim=1).squeeze(-1)

    def export_gaussians_ply(self, gaussians, file_path='output/scene.ply'):
        features, opacities, rotations, scales, xyz = gaussians.features, gaussians.opacity, gaussians.rotation, gaussians.scaling, gaussians.xyz
        features = features.reshape(-1, 3*(self._max_sh_degree +1)**2)
        f_dc, f_rest = features[:, :3], features[:, 3:]
        scales = torch.clamp_min(torch.nan_to_num(scales, nan=1e-5, posinf=5, neginf=1e-5), min=1e-5)
        scales = scales.log()
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = f_dc.detach().cpu().numpy()
        f_rest = f_rest.detach().cpu().numpy()
        opacities = opacities.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations=rotations.detach().cpu().numpy()
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
        attr = self.construct_list_of_attributes(f_dc, f_rest, scales, rotations)
        dtype_full = [(attribute, 'f4') for attribute in attr]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        if os.path.isfile(file_path):
            os.remove(file_path)

        PlyData([el]).write(file_path)


    def param_count_report(self):
        sh_rest = self.spherical_harmonics_rest.flatten().size()[0]
        sh_dc = self.spherical_harmonics_dc.flatten().size()[0]
        scale_params_count = self.scale_params.flatten().size()[0]
        q_params_count = self.quaternion_params.flatten().size()[0]
        cps = self.control_points.flatten().size()[0]

        num_elements = cps + sh_dc + sh_rest + q_params_count + scale_params_count
        print(f"Total params to optimize: {num_elements}")

    ###########################################################################
    ############################ Getters / Setters ############################
    ###########################################################################

    @property
    def get_features(self):
        return self.gaussians.features

    @property
    def get_control_points(self):
        return self.control_points

    @property
    def get_device(self):
        return self.device

    @property
    def get_scaling(self):
        return self.scaling_activation(self.gaussians.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.gaussians.rotation)

    @property
    def get_xyz(self):
        return self.gaussians.xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self.gaussians.opacity)

    def get_num_patches(self):
        return self.num_patches

    def get_patches(self):
        return self._patches

    def get_degrees_u(self):
        return self._degrees_u

    def get_degrees_v(self):
        return self._degrees_v

    def get_control_points(self):
        return self.control_points

    def get_knots_u(self):
        return self._knots_u

    def get_knots_v(self):
        return self._knots_v

    def get_res(self):
        return self._res

    def get_db_u_tensor(self):
        return self.__db_u_tensor

    def get_db_v_tensor(self):
        return self.__db_v_tensor

    def get_surface_points(self):
        return self.surface_points

    def get_surface_normals(self):
        return self.surface_normals

    def get_derivative_points_u(self):
        return self._du

    def get_derivative_points_v(self):
        return self._dv

    def get_debug(self):
        return self.__debug

    def set_device(self, device):
        self.device = device

    def set_patches(self, patches):
        self._patches = patches

    def set_degrees_u(self, degrees_u):
        self._degrees_u = degrees_u

    def set_degrees_v(self, degrees_v):
        self._degrees_v = degrees_v

    def set_control_points(self, control_points):
        self.control_points = control_points

    def set_knots_u(self, knots_u):
        self._knots_u = knots_u

    def set_knots_v(self, knots_v):
        self._knots_v = knots_v

    def set_res(self, res):
        self._res = res

    def set_debug(self, debug):
        self.__debug = debug

################################################################################################
#############################################OLDER CODE#########################################
################################################################################################
