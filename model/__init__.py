import gc

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyElement, PlyData
from torch.nn import functional as F

from utils.general_utils import get_expon_lr_func_with_restarts_and_decay, plot_lr_schedule
from .schedulers import DecayingCosineAnnealingWarmRestarts
from .spline_utils import b_spline_basis_functions_and_derivatives, \
    normalize_point_cloud, \
    inverse_sigmoid, Gaussians, quaternion_from_two_vectors, analyze_gradient_trend_per_patch, patch_subdivisions, SH_interpolation, \
    catmull_clark_subdivision, evaluate_bspline_surface
import os

RESCALING_UP = 1.
RESCALING_DOWN = .9


class SplineModel(nn.Module):
    def __init__(self, patches, config):
        super(SplineModel, self).__init__()
        self.step_size = config.step_size
        self.split_every = config.split_every
        self.config = config
        self.device = config.device
        num_patches = len(patches)
        self.num_patches = num_patches
        self.__debug = config.debug
        self._res = config.resolution
        self._max_sh_degree = config.max_sh_degree
        self.active_sh_degree = 0
        ctrl_pts = normalize_point_cloud(torch.stack([patch[2] for patch in patches]))
        ctrl_pts = ctrl_pts.flatten(end_dim=-2).unique(dim=-2)

        noise = torch.randn_like(ctrl_pts, requires_grad=False) * 0.
        self.control_points = nn.Parameter(ctrl_pts + noise, requires_grad=True)

        sph_harm_features = torch.ones(
            (self.num_patches, ctrl_pts.shape[1], ctrl_pts.shape[2], (self._max_sh_degree + 1) ** 2, 3), device=config.device).contiguous()
        self.spherical_harmonics_dc = nn.Parameter(sph_harm_features[:, :, :, :1, :], requires_grad=True)
        self.spherical_harmonics_rest = nn.Parameter(torch.zeros_like(sph_harm_features[:, :, :, 1:, :]), requires_grad=True)
        self.scale_params = nn.Parameter(torch.ones((num_patches, 2), device=config.device), requires_grad=True)

        # self.quaternion_params = nn.Parameter(torch.ones((num_patches, 4), device=device), requires_grad=False)
        self.nurbs_weights = nn.Parameter(
            torch.ones((self.num_patches, self.control_points.shape[1], self.control_points.shape[2]),
                       dtype=torch.float32, device=config.device, requires_grad=False))

        # learning rates
        self.position_lr = config.position_lr
        self.feature_lr = config.feature_lr
        self.scale_params_lr = config.scale_params_lr
        self.nurbs_weights_lr = config.nurbs_weights_lr
        self.position_lr_final = config.position_lr_final
        self.feature_lr_final = config.feature_lr_final
        self.scale_params_lr_final = config.scale_params_lr_final
        self.nurbs_weights_lr_final = config.nurbs_weights_lr_final

        # Variables to store computation results
        self._patches = patches
        self._degrees_u = torch.empty(0)
        self._degrees_v = torch.empty(0)
        self._knots_u = torch.empty(0)
        self._knots_v = torch.empty(0)
        self.surface_points = torch.empty(0)
        self.surface_normals = torch.empty(0)

        # Other variables
        self.cooldown_patches = torch.zeros(num_patches, dtype=bool, device=config.device)

        # For 3D Gaussians post-processing sampling
        self.gaussians = torch.empty(0)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self._opacity = torch.ones((num_patches * config.resolution**2, 1), requires_grad=False, device=config.device)

        self.set_surface_data()
        self.training_setup()
        # self.control_points_upsampling(torch.ones(self.num_patches, dtype=bool, device=self.device))

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
        self.areas = torch.empty((self.num_patches, 1), device=self.device, dtype=torch.float32, requires_grad=True)
        self.control_points_grads = torch.zeros((self.split_every // self.step_size, self.control_points.shape[0], self.control_points.shape[1], self.control_points.shape[2], self.control_points.shape[3]), device=self.device)
        self.spherical_harmonics_dc_grads = torch.zeros((self.split_every // self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], 1, 3), device=self.device)
        self.spherical_harmonics_rest_grads = torch.zeros((self.split_every // self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2 - 1, 3), device=self.device)
        self._opacity = torch.ones((self.num_patches * self._res * self._res, 1), requires_grad=True,
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
        knots_u = self._knots_u[patch_id]
        knots_v = self._knots_v[patch_id]
        u_values = torch.linspace(knots_u[degree_u], knots_u[-degree_u - 1], num_points_u)
        v_values = torch.linspace(knots_v[degree_v], knots_v[-degree_v - 1], num_points_v)
        self.b_u_tensor[patch_id], self.db_u_tensor[patch_id],  self.ddb_u_tensor[patch_id] = (
            b_spline_basis_functions_and_derivatives(degree_u, knots_u.cpu(), u_values, device))
        self.b_v_tensor[patch_id], self.db_v_tensor[patch_id],  self.ddb_v_tensor[patch_id] = (
            b_spline_basis_functions_and_derivatives(degree_v, knots_v.cpu(), v_values, device))
        self.sh_features = torch.concatenate([self.spherical_harmonics_dc, self.spherical_harmonics_rest], dim=-2)


    def fit_nurbs(self):
        epsilon = 10 # Small value to prevent division by zero
        denominator = torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.b_v_tensor, self.nurbs_weights).flatten(start_dim=0).unsqueeze(-1)
        denominator[denominator == 0] = 1

        # denominator
        weighted_control_points = self.control_points * self.nurbs_weights.unsqueeze(-1)


        self.xyz = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, weighted_control_points).view(-1, 3) / denominator


        self.dus = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.b_v_tensor,
                                            weighted_control_points).reshape(-1, 3) / denominator

        self.dvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.db_v_tensor,
                                            weighted_control_points).reshape(-1, 3) / denominator

        self.ddus = torch.einsum('aik,ajl,aklm->aijm', self.ddb_u_tensor, self.b_v_tensor,
                           weighted_control_points).reshape(-1, 3) / denominator

        self.ddvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.ddb_v_tensor,
                            weighted_control_points).reshape(-1, 3) / denominator


        self.dduvs = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.db_v_tensor,
                            weighted_control_points).reshape(-1, 3) / denominator

        # SH sampling
        sh_features = torch.concatenate([self.spherical_harmonics_dc, self.spherical_harmonics_rest], dim=-2)
        SHs_sampling = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, sh_features.view(self.num_patches, 4, 4, -1))

        ################################
        ########### Scaling ###########
        ################################

        # Compute patch areas and bounding box volumes

        # Compute scaling factors based on patch area, bounding box volume, and curvature
        scale_params = torch.cat([self.scale_params, torch.ones((self.scale_params.shape[0], 1), dtype=torch.float32, device=self.device, requires_grad=True)*1e-16], dim=-1)

        scale_factors = ((self.compute_patches_area().unsqueeze(-1) / (self._res ** 2)) * scale_params).repeat_interleave(self._res ** 2, dim=0)

        # Compute the normal vector (n = r_u x r_v)
        self.surface_normals = F.normalize(torch.cross(self.dus, self.dvs, dim=-1),  p=1, dim=-1)

        # Determine the scale for each axis (x, y, z) for the Gaussians
        scaling_x = (self.dus * self.dus).sum(-1) + epsilon
        scaling_y = (self.dvs * self.dvs).sum(-1) + epsilon
        scaling_z = epsilon * torch.ones_like(scaling_x)

        # Combine the scales into a single tensor
        scaling = torch.stack([scaling_x, scaling_y, scaling_z], dim=-1)
        cx = (self.ddus).sum(-1)
        cy = (self.ddvs).sum(-1)
        cz = torch.ones_like(cx)

        # Combine the scales into a single tensor
        curvature = torch.stack([cx, cy, cz], dim=-1)

        scaling = ((scale_factors * scaling).abs().sqrt().log() - (curvature.abs() + epsilon).log())

        # Compute rotation
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.xyz.shape[0], 3)
        rotation = quaternion_from_two_vectors(up, self.surface_normals.reshape(-1, 3)) #* self.quaternion_params.repeat_interleave(self._res ** 2, dim=0)

        # Reshape surface normals
        self.surface_normals = self.surface_normals.view(self.num_patches, self._res, self._res, 3)

        return self.xyz, self._opacity, scaling, rotation, SHs_sampling.reshape(-1, (self._max_sh_degree + 1) ** 2, 3)


    ########################################################################
    ############################# DENSIFICATION ############################
    ########################################################################

    def visibility_filter_to_patches(self, visibility_filter: torch.Tensor):
        mask = visibility_filter.reshape(self.num_patches, -1).sum(dim=-1)
        mask[mask < (self._res**2)*0.25] = 0
        return mask.type(torch.BoolTensor)


    def grads_handler(self, iteration, visibility_filter, grad_clipper=.0, retain_grad=False):
        mask = self.visibility_filter_to_patches(visibility_filter)
        # Compute the gradient of self.control_points
        pos = (iteration//self.step_size - 1 ) % (self.split_every // self.step_size)
        # self.control_points_grads[pos][mask] = self.control_points.grad[mask] / self.splitting_interval_every
        self.control_points_grads[pos][mask] = torch.norm(self.control_points.grad[mask], dim=-1, keepdim=True) / self.split_every
        # Compute the gradient of self.spherical_harmonics_dc
        # self.spherical_harmonics_dc_grads[pos][mask] = self.spherical_harmonics_dc.grad[mask] / self.splitting_interval_every
        self.spherical_harmonics_dc_grads[pos][mask] = torch.norm(self.spherical_harmonics_dc.grad[mask], dim=-1, keepdim=True) / self.split_every
        # Compute the gradient of self.spherical_harmonics_rest
        # self.spherical_harmonics_rest_grads[pos][mask] = self.spherical_harmonics_rest.grad[mask] / self.splitting_interval_every
        self.spherical_harmonics_rest_grads[pos][mask] = torch.norm(self.spherical_harmonics_rest.grad[mask], dim=-1, keepdim=True) / self.split_every

        if grad_clipper:
            torch.nn.utils.clip_grad_norm_(self.control_points, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.spherical_harmonics_dc, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.spherical_harmonics_rest, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.scale_params, max_norm=grad_clipper)
            torch.nn.utils.clip_grad_norm_(self.nurbs_weights, max_norm=grad_clipper)

        if retain_grad:
            self.control_points.retain_grad()
            self.spherical_harmonics_dc.retain_grad()
            self.spherical_harmonics_rest.retain_grad()
            self.scale_params.retain_grad()
            self.nurbs_weights.retain_grad()


    def patches_to_split(self):
        # Shape is (BATCH x Patches x 4 x 4 x 3) representing a batch of patches of a 3D surface
        areas_normalizer = F.softmax(self.areas, dim=0).unsqueeze(-1)
        control_points_grads = self.control_points_grads
        areas_normalizer = torch.ones_like(areas_normalizer.unsqueeze(0))

        control_points_converging = analyze_gradient_trend_per_patch(control_points_grads * areas_normalizer, param_group_name='control_points', lr=self.position_lr)
        # Shape is (BATCH x Patches x 4 x 4 x Max_shperical_harmonic_coefficients x 3) representing a SH features corresponds to each control point
        spherical_harmonics_dc_grads = (self.spherical_harmonics_dc_grads * areas_normalizer.unsqueeze(-1)).reshape(self.split_every // self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)
        spherical_harmonics_rest_grads = (self.spherical_harmonics_rest_grads * areas_normalizer.unsqueeze(-1)).reshape(self.split_every // self.step_size, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)
        if not self.active_sh_degree:
            SH_grads = analyze_gradient_trend_per_patch(spherical_harmonics_dc_grads, param_group_name='f_dc', lr=self.feature_lr)
        else:
            SH_grads = analyze_gradient_trend_per_patch(spherical_harmonics_rest_grads, param_group_name='f_rest', lr=self.feature_lr/20)
        SH_mask = SH_grads['diverging']
        return SH_mask & control_points_converging['diverging'] & (~self.cooldown_patches)

    def patch_upsampler(self):
        with torch.no_grad():
            print(f"\nnum patches before: {self.num_patches}")
            patch_to_split_indices = self.patches_to_split()
            self.control_points_upsampling(patch_to_split_indices)
            print(f"num patches after: {self.num_patches}\n")

    def control_points_upsampling(self, do_upsampling_mask):
        non_zeros = do_upsampling_mask.sum().item()
        features = torch.cat([self.spherical_harmonics_dc.clone().detach(), self.spherical_harmonics_rest.clone().detach()], dim=-2).reshape(self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2, 3)
        features_to_optimize = features[do_upsampling_mask]

        control_points_to_optimize = self.control_points[do_upsampling_mask].clone().detach()
        features_other = features[~do_upsampling_mask]
        old_control_points = self.control_points[~do_upsampling_mask]

        # Prepare updated parameters values
        splitted_control_points = catmull_clark_subdivision(control_points_to_optimize)
        try:
            new_sh = catmull_clark_subdivision(features_to_optimize.flatten(start_dim=-2)).reshape(do_upsampling_mask.sum()*4, 4, 4, -1, 3)
            new_sh[:,:,:, self.active_sh_degree] = new_sh[:,:,:, self.active_sh_degree] # + torch.randn_like(new_sh[:,:,:, self.active_sh_degree, 0].unsqueeze(-1))
        except RuntimeError as _:
            return
        # new_sh =SH_interpolation(features_to_optimize)
        # splitted_control_points = patch_subdivisions(control_points_to_optimize)
        new_control_points = torch.cat((old_control_points, splitted_control_points)).contiguous()
        new_nurbs_weights = torch.cat((self.nurbs_weights[~do_upsampling_mask], self.nurbs_weights[do_upsampling_mask].repeat_interleave(4, dim=0))).contiguous()
        sph_harm_features = torch.cat((features_other, new_sh))
        new_features_dc = sph_harm_features[:, :, :, : 1, :].contiguous()
        new_features_rest = sph_harm_features[:, :, :, 1:, :].contiguous()
        self.cooldown_patches = torch.cat((
            torch.zeros(self.num_patches - do_upsampling_mask.sum(), dtype=bool, device=self.device),
            torch.ones(4*do_upsampling_mask.sum(), device=self.device, dtype=bool)))

        if self.num_patches == do_upsampling_mask.sum():
            self.cooldown_patches = torch.zeros_like(self.cooldown_patches)
        if non_zeros > 0:
            existing_scaling = self.scale_params[~do_upsampling_mask]
            splitted_scaling = self.scale_params[do_upsampling_mask].repeat_interleave(4, dim=0) * RESCALING_UP
            new_scaling = torch.cat([existing_scaling, splitted_scaling], dim=0).contiguous()

        else:
            new_scaling = self.scale_params[~do_upsampling_mask]

        # From Gaussian Splatting
        d = {"control_points": new_control_points,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "scale_params": new_scaling,
             "nurbs_weights": new_nurbs_weights
             }

        optimizable_tensors = self.override_tensors_in_optimizer(d)
        self.control_points = optimizable_tensors["control_points"]
        self.spherical_harmonics_dc = optimizable_tensors["f_dc"]
        self.spherical_harmonics_rest = optimizable_tensors["f_rest"]
        self.scale_params = optimizable_tensors["scale_params"]
        self.nurbs_weights = optimizable_tensors["nurbs_weights"]
        self.num_patches = len(self.control_points)
        self._patches = [(self._degrees_u[0], self._degrees_v[0], self.control_points[patchId]) for patchId in range(self.num_patches)]
        self.free_cuda_memory()
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
            {'params': self.spherical_harmonics_rest, 'lr': self.feature_lr / 20, 'min_lr': self.feature_lr_final / 20, "name": "f_rest"},
            {'params': self.scale_params,  'lr': self.scale_params_lr, 'min_lr': self.scale_params_lr_final,"name": "scale_params"},
            {'params': self.nurbs_weights, 'lr': self.nurbs_weights_lr, 'min_lr': self.nurbs_weights_lr_final, "name": "nurbs_weights"}
        ]
        max_steps = 30_000

        self.optimizer = torch.optim.Adam(self.param_groups, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func_with_restarts_and_decay(lr_init=self.position_lr,
                                                                            lr_final=self.position_lr_final,
                                                                            max_steps=max_steps,
                                                                            restart_steps=int(
                                                                                self.split_every * 1.5),
                                                                            decay_factor=self.config.decay_factor, fine_tune_from=self.config.stop_splitting)


        self.scaling_scheduler_args = get_expon_lr_func_with_restarts_and_decay(lr_init=self.scale_params_lr,
                                                                                lr_final=self.scale_params_lr_final,
                                                                                max_steps=max_steps,
                                                                                restart_steps=self.split_every,
                                                                                decay_factor=self.config.decay_factor, fine_tune_from=self.config.stop_splitting)


        self.feature_lr_args = get_expon_lr_func_with_restarts_and_decay(lr_init=self.feature_lr,
                                                                         lr_final=self.feature_lr_final,
                                                                         max_steps=max_steps,
                                                                         restart_steps=self.split_every,
                                                                         decay_factor=self.config.decay_factor, fine_tune_from=self.config.stop_splitting)

        plot_lr_schedule(self.xyz_scheduler_args, M=max_steps)
        plot_lr_schedule(self.scaling_scheduler_args, M=max_steps)
        plot_lr_schedule(self.feature_lr_args, M=max_steps)
        self.scheduler_args = {"control_points": self.xyz_scheduler_args,
                          "f_dc" : self.feature_lr_args,
                          "f_rest" : self.feature_lr_args,
                          "scale_params" : self.scaling_scheduler_args,
                          # "quaternion_params" : self.quaternion_scheduler_args
                          }


    def step(self, iteration, visibility_filter):
        self.grads_handler(iteration, visibility_filter, retain_grad=False, grad_clipper=0)
        for param_group in self.optimizer.param_groups:
            try:
                lr = self.scheduler_args[param_group["name"]](iteration)
                param_group['lr'] = lr
            except KeyError:
                continue
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


    def override_tensors_in_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            new_tensor = tensors_dict[group["name"]]  # This is the new tensor to replace the existing one
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                # Resetting optimizer states (assuming momentum buffers etc. should be reinitialized)
                stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)

                # Remove old state references
                del self.optimizer.state[group['params'][0]]

                # Replace the parameter tensor
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                # Simply replace the tensor if there was no stored state
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            if group['name'] == 'f_rest' or group['name'] == 'scale_params':
                group['lr'] = group['lr']
            optimizable_tensors[group["name"]] = group["params"][0]
        gc.collect()
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
        # q_params_count = self.quaternion_params.flatten().size()[0]
        cps = self.control_points.flatten().size()[0]

        num_elements = cps + sh_dc + sh_rest + scale_params_count
        print(f"Total params to optimize: {num_elements}")

    def free_cuda_memory(self):
        # List of tensor attributes to delete
        tensor_attributes = [
            '_degrees_u', '_degrees_v', '_knots_u', '_knots_v',
            'b_u_tensor', 'b_v_tensor', 'db_u_tensor', 'db_v_tensor',
            'ddb_u_tensor', 'ddb_v_tensor', 'dus', 'dvs', 'ddus', 'ddvs',
            'dduvs', 'areas', 'control_points_grads',
            'spherical_harmonics_dc_grads', 'spherical_harmonics_rest_grads',
            '_opacity'
        ]

        # Delete each tensor and free CUDA memory
        for attr in tensor_attributes:
            if hasattr(self, attr):
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    del tensor
                    setattr(self, attr, None)

        # Explicitly call the garbage collector to free unreferenced memory
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        print("CUDA memory for specified tensors has been freed.")

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

    @property
    def get_dus(self):
        return self.dus.reshape(self.num_patches, self._res, self._res, 3)
    @property
    def get_ddus(self):
        return self.ddus.reshape(self.num_patches, self._res, self._res, 3)

    @property
    def get_dvs(self):
        return self.dvs.reshape(self.num_patches, self._res, self._res, 3)

    @property
    def get_ddvs(self):
        return self.ddvs.reshape(self.num_patches, self._res, self._res, 3)

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
