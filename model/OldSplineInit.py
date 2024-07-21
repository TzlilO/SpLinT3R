import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from plyfile import PlyElement, PlyData
from torch.nn import functional as F
from .spline_utils import to_quaternions, b_spline_basis_functions_and_derivatives, patch_subdivision, \
    patches_subdivision, SH_interpolation, \
    normalize_point_cloud, compute_scale_factors, calculate_convex_hull_volume, compute_bounding_box_volume
from utils.general_utils import get_expon_lr_func
from utils.viz import plot_b_spline_surface_with_derivatives
import os
from typing import NamedTuple
from simple_knn._C import distCUDA2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class Gaussians(NamedTuple):
    xyz: torch.Tensor
    features: torch.Tensor
    scaling: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor
    active_sh_degree: int = 1

class SplineSurface(NamedTuple):
    surface_points: torch.Tensor
    du: torch.Tensor
    dv: torch.Tensor
    dduv: torch.Tensor
    dduu: torch.Tensor
    ddvv: torch.Tensor
    curvature_gaussian: torch.Tensor
    curvature_mean: torch.Tensor


class SplineModel(nn.Module):
    def __init__(self, patches, device='cuda', res=10, debug=False, splitting_interval_every=1000):
        super(SplineModel, self).__init__()
        self.splitting_interval_every = splitting_interval_every
        self.active_sh_degree = 0
        self.spatial_lr_scale = 5.5
        self.device = device
        num_patches = len(patches)
        self.num_patches = num_patches
        self.__debug = debug
        self._res = res
        self._max_sh_degree = 6
        ctrl_pts = normalize_point_cloud(torch.stack([patch[2] for patch in patches]))

        noise = torch.randn_like(ctrl_pts, requires_grad=True) * 0.1
        self.control_points = nn.Parameter(ctrl_pts, requires_grad=True)

        sph_harm_features = torch.ones(
            (self.num_patches, ctrl_pts.shape[1], ctrl_pts.shape[2], (self._max_sh_degree + 1) ** 2, 3),device=device).contiguous()
        self.spherical_harmonics_dc = nn.Parameter(sph_harm_features[:, :, :, :1, :], requires_grad=True)
        self.spherical_harmonics_rest = nn.Parameter(sph_harm_features[:, :, :, 1:, :]*.0, requires_grad=True)
        self.scale_params = nn.Parameter(torch.ones((num_patches, 1, 3), device=device), requires_grad=True)
        self.quaternion_params = nn.Parameter(torch.ones((num_patches, 1, 4), device=device), requires_grad=True)

        # learning rates
        self.position_lr = 1e-5
        self.nurbs_weights_lr = 1.6e-6
        self.nurbs_weights_SH_lr = 1.6e-6
        self.feature_lr = 5e-2
        self.scale_params_lr = 1e-3
        self.quaternion_params_lr = 1e-2
        self.position_lr_final = 0.0005
        self.feature_lr_final = 0.0025
        self.scale_params_lr_final = 0.000001
        self.quaternion_params_lr_final = 0.00001

        # Variables to store computation results
        self._patches = patches
        self._degrees_u = torch.empty(0)
        self._degrees_v = torch.empty(0)
        self._knots_u = torch.empty(0)
        self._knots_v = torch.empty(0)
        self.set_surface_data()
        self.surface_points = torch.empty(0)
        self.surface_normals = torch.empty(0)

        # For 3D Gaussians post-processing sampling
        self.gaussians = None
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.__opacity = torch.ones((num_patches * res * res, 1), requires_grad=True, device=self.device)

        self.xyz_gradient_accum = torch.zeros((self.control_points.shape[0], self.control_points.shape[1], self.control_points.shape[2], self.control_points.shape[3]), device='cuda')
        self.SH_dc_gradient_accum = torch.zeros((self.splitting_interval_every, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], 1, 3), device='cuda')
        self.SH_rest_gradient_accum = torch.zeros((self.splitting_interval_every, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2 - 1, 3), device='cuda')
        self.denom = torch.empty(0)

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

        self.nurbs_weights = nn.Parameter(torch.ones((self.num_patches, deg_u , deg_v),
                                                     dtype=torch.float32, device=self.device, requires_grad=True))
        self.nurbs_weights_SH = nn.Parameter(torch.ones((self.num_patches, deg_u , deg_v),
                                                     dtype=torch.float32, device=self.device, requires_grad=True))

        self.b_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.b_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()
        self.db_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.db_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()
        self.ddb_u_tensor = torch.empty((self.num_patches, self._res, deg_u), device=self.device).contiguous()
        self.ddb_v_tensor = torch.empty((self.num_patches, self._res, deg_v), device=self.device).contiguous()

        # TODO: Squeeze the batch dimension and modify the gradients_handler accordingly
        self.control_points_grads = torch.zeros((self.splitting_interval_every, self.control_points.shape[0], self.control_points.shape[1], self.control_points.shape[2], self.control_points.shape[3]), device=self.device)
        self.spherical_harmonics_dc_grads = torch.zeros((self.splitting_interval_every, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], 1, 3), device=self.device)
        self.spherical_harmonics_rest_grads = torch.zeros((self.splitting_interval_every, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2 - 1, 3), device=self.device)
        self.dus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.ddus = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.ddvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        self.dduvs = torch.empty((self.num_patches * self._res * self._res, 3), device=self.device, dtype=torch.float32,requires_grad=True)
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

        self.b_u_tensor[patch_id], self.db_u_tensor[patch_id],  self.ddb_u_tensor[patch_id] = b_spline_basis_functions_and_derivatives(degree_u, knots_u.cpu(), u_values, device)
        self.b_v_tensor[patch_id], self.db_v_tensor[patch_id],  self.ddb_v_tensor[patch_id] = b_spline_basis_functions_and_derivatives(degree_v, knots_v.cpu(), v_values, device)

    def inverse_sigmoid(self, x):
        return torch.log(x / (1 - x))
    def sample_gaussians(self):
        # xyz, opacity, scaling, rotation, features = self.fit_splines()
        xyz, opacity, scaling, rotation, features = self.fit_nurbs()
        self.gaussians = Gaussians(xyz=xyz, opacity=self.inverse_sigmoid(opacity), scaling=scaling, rotation=rotation, features=features)
        return self.gaussians


    def training_setup(self):

        # self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        # self.SH_dc_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.SH_rest_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self.control_points], 'lr': self.position_lr, "name": "control_points"},
            {'params': [self.spherical_harmonics_dc], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self.spherical_harmonics_rest], 'lr': self.feature_lr / 20, "name": "f_rest"},
            {'params': [self.scale_params], 'lr': self.scale_params_lr, "name": "scale_params"},
            {'params': [self.quaternion_params], 'lr': self.quaternion_params_lr, "name": "quaternion_params"},
            {'params': [self.nurbs_weights], 'lr': self.nurbs_weights_lr / 10, "name": "nurbs_weights"},
            {'params': [self.nurbs_weights_SH], 'lr': self.nurbs_weights_SH_lr / 10, "name": "nurbs_weights_SH"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        max_steps=15000
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr,
                                                    lr_delay_steps=3_000,
                                                    lr_final=self.position_lr_final,
                                                    max_steps=max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=self.scale_params_lr,
                                                        lr_delay_steps=3_000,
                                                        lr_final=self.scale_params_lr_final,
                                                        max_steps=max_steps)
        self.quaternion_scheduler_args = get_expon_lr_func(lr_init=self.quaternion_params_lr,
                                                           lr_delay_steps=3_000,
                                                           lr_final=self.quaternion_params_lr_final,
                                                           max_steps=max_steps)

        self.feature_lr_args = get_expon_lr_func(lr_init=self.feature_lr,
                                                    lr_delay_steps=3_000,
                                                    lr_final=self.feature_lr_final,
                                                    max_steps=max_steps)

        self.scheduler_args = {"control_points": self.xyz_scheduler_args,
                          "f_dc" : self.feature_lr_args,
                          "f_rest" : self.feature_lr_args,
                          "scale_params" : self.scaling_scheduler_args,
                          "quaternion_params" : self.quaternion_scheduler_args
                          }

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            try:
                lr = self.scheduler_args[param_group["name"]](iteration)
                param_group['lr'] = lr
            except KeyError:
                continue

            return lr

    def update_named_lr(self, name, factor):
        ''' Learning rate scheduling per step '''
        param_group = self.optimizer.param_group[name]
        new_lr = param_group['lr'] * factor
        param_group['lr'] = new_lr

    ########################################################################
    ############################# DENSIFICATION ############################
    ########################################################################

    def visibility_filter_to_patches(self, visibility_filter: torch.Tensor):
        mask = visibility_filter.reshape(self.num_patches, -1).sum(dim=-1)
        mask[mask < self._res**2 / 2] = 0
        return mask.type(torch.BoolTensor)


    def grads_handler(self, iteration, visibility_filter):
        mask = self.visibility_filter_to_patches(visibility_filter)

        # Compute the gradient of self.control_points
        pos = (iteration-1) % self.splitting_interval_every
        self.control_points_grads[pos][mask] = torch.norm(self.control_points.grad[mask], dim=-1, keepdim=True) / self.splitting_interval_every
        # torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        # Compute the gradient of self.spherical_harmonics_dc
        # self.spherical_harmonics_dc_grads[pos][mask] = self.spherical_harmonics_dc.grad[mask]
        self.spherical_harmonics_dc_grads[pos][mask] = torch.norm(self.spherical_harmonics_dc.grad[mask], dim=-1, keepdim=True) / self.splitting_interval_every

        # Compute the gradient of self.spherical_harmonics_rest
        # self.spherical_harmonics_rest_grads[pos][mask] = self.spherical_harmonics_rest.grad[mask]
        self.spherical_harmonics_rest_grads[pos][mask] = torch.norm(self.spherical_harmonics_rest.grad[mask], dim=-1, keepdim=True) / self.splitting_interval_every


    def patches_to_split(self):
        # denom = 4
        # Shape is (BATCH x Patches x 4 x 4 x 3) representing a batch of patches of a 3D surface
        control_points_grads = self.control_points_grads

        # Shape is (BATCH x Patches x 4 x 4 x Max_shperical_harmonic_coefficients x 3) representing a SH features corresponds to each control point
        spherical_harmonics_dc_grads = self.spherical_harmonics_dc_grads.reshape(self.splitting_interval_every, self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)
        spherical_harmonics_rest_grads = self.spherical_harmonics_dc_grads.reshape(self.splitting_interval_every,self.num_patches, self.control_points.shape[1], self.control_points.shape[2], -1, 3)

        # Sum the gradients across all batches [(BATCH x Patches x 4 x 4 x 3) ------------>  (Patches)]
        accum_cp = control_points_grads.abs().sum(axis=0).flatten(start_dim=1).sum(axis=1)
        accum_sh_dc = spherical_harmonics_dc_grads.sum(axis=0).flatten(start_dim=1).sum(axis=1)
        accum_sh_rest = spherical_harmonics_rest_grads.sum(axis=0).flatten(start_dim=1).sum(axis=1)

        total_accum_grads = (accum_cp + accum_sh_dc + accum_sh_rest) / 3.0
        selected_pts_mask = torch.where(total_accum_grads > torch.quantile(total_accum_grads, q=0.9), True, False)

        # highest_grads_mask = torch.zeros(total_accum_grads.shape, dtype=torch.int64, device=self.device)
        # highest_grads = torch.topk(total_accum_grads, k=int(len(total_accum_grads)/denom)).indices
        # highest_grads_mask.scatter_(0, highest_grads, torch.ones_like(highest_grads))


        # Calculate cosine similarity between consequent saved grads [(BATCH x Patches x 4 x 4 x 3) ------------>  (Patches)]
        # If the cosine similarity is greater than a threshold, the two patches are candidates for splitting.
        # cos_sim_cp = F.cosine_similarity(control_points_grads[:-1], control_points_grads[1:], dim=0).flatten(start_dim=1).sum(axis=-1)
        # cos_sim_dc = F.cosine_similarity(spherical_harmonics_dc_grads[:-1], spherical_harmonics_dc_grads[1:], dim=0).flatten(start_dim=1).sum(axis=-1)
        # cos_sim_rest = F.cosine_similarity(spherical_harmonics_rest_grads[:-1], spherical_harmonics_rest_grads[1:], dim=0).flatten(start_dim=1).sum(axis=-1)
        # total_cos_sim = cos_sim_cp * cos_sim_dc * cos_sim_rest
        # poor_rec = torch.zeros(total_cos_sim.shape, dtype=torch.int64, device=self.device)
        # are_poor = torch.topk(total_cos_sim * -1, k=int(len(total_cos_sim)/denom)).indices
        # poor_rec.scatter_(0, are_poor, torch.ones_like(are_poor))

        # Compute the variance of the gradients
        # Sum the gradients across all batches [(BATCH x Patches x 4 x 4 x 3) ------------>  (Patches)]
        # var_scores_cp = torch.var(control_points_grads, dim=0).flatten(start_dim=1).sum(axis=-1)
        # var_scores_dc = torch.var(spherical_harmonics_dc_grads, dim=0).flatten(start_dim=1).sum(axis=-1)
        # var_scores_rest = torch.var(spherical_harmonics_rest_grads, dim=0).flatten(start_dim=1).sum(axis=-1)
        # total_variance = var_scores_cp + var_scores_dc * var_scores_rest
        # top_k_non_convergence = torch.topk(total_variance, k=int(len(total_variance)/denom)).indices
        # non_conv = torch.zeros(total_variance.shape, dtype=torch.int64, device=self.device)
        # non_conv.scatter_(0, top_k_non_convergence, torch.ones_like(top_k_non_convergence))
        # positive_mask = (torch.tensor(poor_rec, dtype=bool)) & (torch.tensor(highest_grads_mask, dtype=bool))
        return selected_pts_mask

    def patch_upsampler(self):
        print(f"num patches before: {self.num_patches}")
        patch_to_split_indices = self.patches_to_split()
        self.control_points_upsampling(patch_to_split_indices)
        print(f"num patches after: {self.num_patches}")

    def control_points_upsampling(self, do_upsampling_mask):
        non_zeros = torch.count_nonzero(do_upsampling_mask).item()
        features = torch.cat([self.spherical_harmonics_dc.clone().detach(), self.spherical_harmonics_rest.clone().detach()], dim=-2).reshape(self.num_patches, self.control_points.shape[1], self.control_points.shape[2], (self._max_sh_degree + 1) ** 2, 3)
        features_to_optimize = features[do_upsampling_mask]
        control_points_to_optimize = self.control_points[do_upsampling_mask].clone().detach()
        features_other = features[~do_upsampling_mask]


        old_control_points = self.control_points[~do_upsampling_mask]
        # Prepare updated parameters values
        new_sh, splitted_control_points = SH_interpolation(control_points_to_optimize, features_to_optimize, target_shape=(7, 7))
        new_control_points = torch.cat((old_control_points, splitted_control_points)).contiguous()
        new_nurbs_weights = torch.cat((self.nurbs_weights[~do_upsampling_mask], torch.ones_like(splitted_control_points[:, :, :,0]))).contiguous()
        new_nurbs_weights_SH = torch.cat((self.nurbs_weights_SH[~do_upsampling_mask], torch.ones_like(splitted_control_points[:, :, :,0]))).contiguous()
        sph_harm_features = torch.cat((features_other, new_sh))
        new_features_dc = sph_harm_features[:,:,:, :1, :].contiguous()
        new_features_rest = sph_harm_features[:,:,:, 1:, :].contiguous()
        if non_zeros:
            # alpha = 1  # smaller values will scale down the
            # beta = 1.01  # Higher values correspond to scaling down the splitted patches scaling factors
            # old_dist2 = (1 - distCUDA2(old_control_points)/2)
            # new_dist2 = (1 - distCUDA2(splitted_control_points)) ** 2
            # new_scaling = torch.cat([existing_scaling.squeeze(1) + old_dist2.unsqueeze(1), splitted_scaling.squeeze(1) * new_dist2.unsqueeze(1)], dim=0).view(-1, 1, 3)
            existing_scaling = self.scale_params[~do_upsampling_mask]
            splitted_scaling = self.scale_params[do_upsampling_mask].clone().detach() * 1.000000005
            splitted_scaling = torch.cat([splitted_scaling, splitted_scaling,splitted_scaling,splitted_scaling])
            new_scaling = torch.cat([existing_scaling.squeeze(1), splitted_scaling.squeeze(1)], dim=0).contiguous().reshape(-1, 1, 3)
            new_rotation = torch.cat([self.quaternion_params[~do_upsampling_mask], self.quaternion_params[do_upsampling_mask].expand((non_zeros, 4, 4)).contiguous().reshape(-1, 1, 4)], dim=0)
        else:
            new_scaling = self.scale_params[~do_upsampling_mask]
            new_rotation = self.quaternion_params[~do_upsampling_mask]

        # From Gaussian Splatting
        d = {"control_points": new_control_points,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "scale_params": new_scaling,
             "quaternion_params": new_rotation,
             "nurbs_weights": new_nurbs_weights,
             "nurbs_weights_SH": new_nurbs_weights_SH
             }

        optimizable_tensors = self.override_tensors_in_optimizer(d)
        self.control_points = optimizable_tensors["control_points"]
        self.spherical_harmonics_dc = optimizable_tensors["f_dc"]
        self.spherical_harmonics_rest = optimizable_tensors["f_rest"]
        self.scale_params = optimizable_tensors["scale_params"]
        self.quaternion_params = optimizable_tensors["quaternion_params"]
        self.nurbs_weights = optimizable_tensors["nurbs_weights"]
        self.nurbs_weights_SH = optimizable_tensors["nurbs_weights_SH"]

        self.num_patches = len(self.control_points)
        self._patches = [(self._degrees_u[0], self._degrees_v[0], self.control_points[patchId]) for patchId in range(self.num_patches)]
        self.set_surface_data()

        sh_rest = self.spherical_harmonics_rest.flatten().size()[0]
        sh_dc = self.spherical_harmonics_dc.flatten().size()[0]
        scale_params_count = self.scale_params.flatten().size()[0]
        q_params_count = self.quaternion_params.flatten().size()[0]
        cps = self.control_points.flatten().size()[0]
        self.__opacity = torch.ones((self.num_patches * self._res * self._res, 1), requires_grad=True, device=self.device)
        num_elements = cps + sh_dc + sh_rest + q_params_count + scale_params_count
        print(f"Total params to optimize: {num_elements}")

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

            optimizable_tensors[group["name"]] = group["params"][0]
        torch.cuda.empty_cache()

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


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

    def control_points_upsampling_iterative(self):
        # TODO: Make sure we're able to densify control points. Afterwards, it should be triggered for patches with poor convergence
        new_control_points = []
        new_sph_harm_features = []
        features = torch.cat([self.spherical_harmonics_dc, self.spherical_harmonics_rest], dim=-2).reshape(
            self.num_patches, self.control_points.shape[1], self.control_points.shape[2],
            (self._max_sh_degree + 1) ** 2 * 3)
        for patchId in range(self.num_patches):
            new_control_points.append(patch_subdivision(self.control_points[patchId]))
            new_sph_harm_features.append(patch_subdivision(features[patchId]))

        # self.control_points = nn.Parameter(torch.stack(new_control_points).to(self.device).reshape(-1, 4, 4, 3), requires_grad=True)
        self.num_patches = len(self.control_points)
        self._patches = [(self._degrees_u[0], self._degrees_v[0], self.control_points[patchId]) for patchId in
                         range(self.num_patches)]
        sph_harm_features = torch.stack(new_sph_harm_features, dim=-2).to(self.device).reshape(
            self.num_patches * self.control_points.shape[1] * self.control_points.shape[2],
            (self._max_sh_degree + 1) ** 2, 3)

        new_features_dc = sph_harm_features[:, :1, :].contiguous()
        new_features_rest = sph_harm_features[:, 1:, :].contiguous()
        new_scaling = self.scale_params.data / 4.
        new_rotation = self.quaternion_params.data / 4.

        # From Gaussian Splatting
        d = {"control_points": torch.stack(new_control_points),
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "scale_params": new_scaling,
             "quaternion_params": new_rotation
             }

        optimizable_tensors = self.override_tensors_in_optimizer(d)
        self.control_points = optimizable_tensors["control_points"]
        self.spherical_harmonics_dc = optimizable_tensors["f_dc"]
        self.spherical_harmonics_rest = optimizable_tensors["f_rest"]
        self.scale_params = optimizable_tensors["scale_params"]
        self.quaternion_params = optimizable_tensors["quaternion_params"]
        self.set_surface_data()

    ###########################################################################
    ############################ Getters / Setters ############################
    ###########################################################################



    @property
    def get_features(self):
        return self.gaussians.features

    @property
    def get_control_points(self):
        return self.control_points
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


    ########################################################################
    ########################## UTILITIES ###################################
    ########################################################################

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



    def fit_splines(self):
        # Up-Sample positions as 3D points
        xyz = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, self.control_points).view(-1, 3)


        dus = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.b_v_tensor,
                                            self.control_points).reshape(-1, 3)

        dvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.db_v_tensor,
                                            self.control_points).reshape(-1, 3)

        ddus = torch.einsum('aik,ajl,aklm->aijm', self.ddb_u_tensor, self.b_v_tensor,
                           self.control_points).reshape(-1, 3)

        ddvs = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.ddb_v_tensor,
                            self.control_points).reshape(-1, 3)


        dduvs = torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.db_v_tensor,
                            self.control_points).reshape(-1, 3)

        # SH sampling
        sh_features = torch.concatenate(
            [self.spherical_harmonics_dc,
             self.spherical_harmonics_rest], dim=1)
        SHs_sampling = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, sh_features.view(self.num_patches, 4, 4, -1))

        scale_factors = compute_scale_factors(dus, dvs)
        # Add epsilon to the norm calculations to avoid division by zero
        epsilon = 1e-3  # Small value to prevent division by zero

        # Compute the first fundamental form coefficients
        E = torch.clamp(torch.sum(dus * dus, dim=-1), min=epsilon)
        F = torch.clamp(torch.sum(dus * dvs, dim=-1), min=epsilon)
        G = torch.clamp(torch.sum(dvs * dvs, dim=-1), min=epsilon)

        # Compute the normal vector (n = r_u x r_v)
        normal_vectors = torch.cross(dus, dvs, dim=-1)
        normal_vectors = normal_vectors / (torch.norm(normal_vectors, dim=-1, keepdim=True) + epsilon)


        # Compute the second fundamental form coefficients
        L = torch.sum(normal_vectors * ddus, dim=-1)
        M = torch.sum(normal_vectors * ddvs, dim=-1)
        N = torch.sum(normal_vectors * dduvs, dim=-1)

        # Compute the determinants of the first and second fundamental forms
        curvature = (E * N + G * L - 2 * F * M) / (2 * (E * G - F * F) + epsilon)
        curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        # Sample Orientations (dU, dV) of for Gaussian primitives at each point
        rotation = to_quaternions(dus.flatten(end_dim=-2), dvs.flatten(end_dim=-2), adjuster=self.quaternion_params)

        # Sampling (scale, pdf) Curvature of the surface to set Gaussian's scale and density
        curvature_transform = torch.clamp(torch.exp(curvature.flatten().unsqueeze(1)), min=0.,
                                          max=5.)
        scaling = curvature_transform * scale_factors
        # scaling[:, -1] = 0.0
        # assert not torch.isnan(scaling).any(), "NaNs in scaling"
        # assert not torch.isnan(ddvs).any(), "NaNs in ddvs"
        # assert not torch.isnan(dduvs).any(), "NaNs in dduvs"
        # assert not torch.isnan(ddus).any(), "NaNs in ddus"
        # assert not torch.isnan(dvs).any(), "NaNs in dvs"
        # assert not torch.isnan(dus).any(), "NaNs in dus"
        # assert not torch.isnan(xyz).any(), "NaNs in xyz"
        # assert not torch.isnan(curvature_gaussian).any(), "NaNs in curvature_gaussian"
        # assert not torch.isnan(curvature_mean).any(), "NaNs in curvature_mean"
        opacity = torch.ones((xyz.shape[0], 1), requires_grad=True, device=self.device)
        return xyz, opacity, torch.clamp_min(torch.nan_to_num(scaling, nan=1e-5, posinf=5, neginf=1e-5), min=1e-5) * torch.clamp_min(self.scale_params, 0.01), rotation, SHs_sampling.reshape(-1, (self._max_sh_degree + 1)**2, 3)

    def fit_nurbs(self):
        denominator = torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.b_v_tensor, self.nurbs_weights).unsqueeze(-1)
        denominator[denominator == 0] = 1

        # Compute the weighted control points
        weighted_control_points = self.control_points * self.nurbs_weights.unsqueeze(-1)

        # Compute the surface points using NURBS basis functions
        # xyz = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor,
        #                    weighted_control_points) / denominator.unsqueeze(-1)
        #
        # # Compute the first derivatives
        # dus = (torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.b_v_tensor, weighted_control_points) -
        #        xyz * torch.einsum('aik,ajl,akl->aij', self.db_u_tensor, self.b_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1)) / denominator.unsqueeze(-1)
        #
        # dvs = (torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.db_v_tensor, weighted_control_points) -
        #        xyz * torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.db_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1)) / denominator.unsqueeze(-1)
        #
        # # Compute the second derivatives
        # ddus = (torch.einsum('aik,ajl,aklm->aijm', self.ddb_u_tensor, self.b_v_tensor, weighted_control_points) -
        #         2 * dus * torch.einsum('aik,ajl,akl->aij', self.db_u_tensor, self.b_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1) -
        #         xyz * torch.einsum('aik,ajl,akl->aij', self.ddb_u_tensor, self.b_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1)) / denominator.unsqueeze(-1)
        #
        # ddvs = (torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.ddb_v_tensor, weighted_control_points) -
        #         2 * dvs * torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.db_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1) -
        #         xyz * torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.ddb_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1)) / denominator.unsqueeze(-1)
        #
        # dduvs = (torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.db_v_tensor, weighted_control_points) -
        #          dus * torch.einsum('aik,ajl,akl->aij', self.b_u_tensor, self.db_v_tensor, self.nurbs_weights).unsqueeze(-1) -
        #          dvs * torch.einsum('aik,ajl,akl->aij', self.db_u_tensor, self.b_v_tensor, self.nurbs_weights).unsqueeze(-1) -
        #          xyz * torch.einsum('aik,ajl,akl->aij', self.db_u_tensor, self.db_v_tensor, self.nurbs_weights).unsqueeze(
        #             -1)) / denominator.unsqueeze(-1)

        # Up-Sample positions as 3D points
        self.xyz = (torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, weighted_control_points
                           ) / denominator).view(-1, 3)


        self.dus = (torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.b_v_tensor,
                                            weighted_control_points) / denominator).reshape(-1, 3)

        self.dvs = (torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.db_v_tensor,
                                            weighted_control_points) / denominator).reshape(-1, 3)

        self.ddus = (torch.einsum('aik,ajl,aklm->aijm', self.ddb_u_tensor, self.b_v_tensor,
                                            weighted_control_points) / denominator).reshape(-1, 3)

        self.ddvs = (torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.ddb_v_tensor,
                            weighted_control_points) / denominator).reshape(-1, 3)


        self.dduvs = (torch.einsum('aik,ajl,aklm->aijm', self.db_u_tensor, self.db_v_tensor,
                            weighted_control_points) / denominator).reshape(-1, 3)

        # SH sampling
        sh_features = torch.concatenate(
            [self.spherical_harmonics_dc,
             self.spherical_harmonics_rest], dim=-2)

        # weigted_sh_features = sh_features.flatten(end_dim=2) * self.nurbs_weights_SH.view(-1, 1, 1)
        weigted_sh_features = sh_features.flatten(end_dim=2)
        SHs_sampling = torch.einsum('aik,ajl,aklm->aijm', self.b_u_tensor, self.b_v_tensor, weigted_sh_features.view(self.num_patches, 4, 4, -1)) #.reshape(self.num_patches, self._res, self._res, sh_features.shape[-2]*sh_features.shape[-1])


        # Add epsilon to the norm calculations to avoid division by zero
        epsilon = 1 # Small value to prevent division by zero

        # Compute the first fundamental form coefficients
        E = torch.clamp(torch.sum(self.dus * self.dus, dim=-1).abs(), min=epsilon)
        F = torch.clamp(torch.sum(self.dus * self.dvs, dim=-1).abs(), min=epsilon)
        G = torch.clamp(torch.sum(self.dvs * self.dvs, dim=-1).abs(), min=epsilon)

        # Compute the normal vector (n = r_u x r_v)
        UxV = torch.cross(self.dus, self.dvs, dim=-1)
        nUxV = torch.norm(UxV, dim=-1, keepdim=True)

        normal_vectors = UxV / (nUxV + epsilon)
        self.surface_normals = normal_vectors.view(self.num_patches, self._res, self._res, 3)

        # Compute the second fundamental form coefficients
        L = torch.clamp(torch.sum(normal_vectors * self.ddus, dim=-1).abs(), min=epsilon)
        M = torch.clamp(torch.sum(normal_vectors * self.ddvs, dim=-1).abs(), min=epsilon)
        N = torch.clamp(torch.sum(normal_vectors * self.dduvs, dim=-1).abs(), min=epsilon)

        # Compute the determinants of the first and second fundamental forms
        curvature_mean = (E * N + G * L - 2 * F * M) / (2 * (E * G - F * F) + epsilon)
        curvature = (curvature_mean - curvature_mean.min() + epsilon)/(curvature_mean.max() - curvature_mean.min())
        curvature = torch.clamp_min(curvature, min=epsilon) * self._res**2
        scale_factors = compute_bounding_box_volume(self.control_points) * self.compute_patches_area().sqrt()
        log_scaling_transform = (scale_factors.unsqueeze(1) / (curvature.reshape(self.num_patches, -1) + 1)).log()

        scaling = (log_scaling_transform.unsqueeze(2) * self.scale_params).reshape(-1, 3)
        rotation = (to_quaternions(self.dus.flatten(end_dim=-2), self.dvs.flatten(end_dim=-2)).reshape(self.num_patches, -1, 4) * self.quaternion_params).reshape(-1, 4)
        return self.xyz, self.__opacity, scaling, rotation, SHs_sampling.view(-1, (self._max_sh_degree + 1)**2, 3)

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
        area = torch.trapz(torch.trapz(differential_area, dim=2), dim=1)

        return area

    def grad_logger(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.SH_dc_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.SH_rest_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1