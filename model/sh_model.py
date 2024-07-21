import torch
import torch.nn as nn

class SHInterpolationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, device='cuda'):
        super(SHInterpolationMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim, device=device), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, output_dim, device=device))
        self.model = nn.Sequential(*layers)

    def forward(self, x, deg):
        return self.model(x)


def prepare_inputs(control_points, sh_coefficients, normals=None, curvatures=None):
    B, P, Q, C = control_points.shape  # B: batch size, P: 4, Q: 4, C: 3
    _, _, _, SH_dim, SH_channels = sh_coefficients.shape  # SH_dim: (1 + SH_deg_max)**2, SH_channels: 3

    inputs = [control_points.view(B, -1, C), sh_coefficients.view(B, -1, SH_dim * SH_channels)]

    # if normals is not None:
    #     inputs.append(normals.view(B, -1, C))
    #
    # if curvatures is not None:
    #     inputs.append(curvatures.view(B, -1, 1))

    combined_inputs = torch.cat(inputs, dim=-1)
    return combined_inputs


def interpolate_SH(control_points, sh_coefficients, mlp, resolution, normals, curvatures,
                   device='cuda'):
    # Prepare inputs for the MLP
    combined_inputs = prepare_inputs(control_points, sh_coefficients, normals, curvatures)
    B, num_samples, input_dim = combined_inputs.shape
    SH_dim = sh_coefficients.shape[-2]
    SH_channels = sh_coefficients.shape[-1]
    output_dim = SH_dim * SH_channels

    mlp_output = mlp(combined_inputs)
    mlp_output = mlp_output.view(B, resolution, resolution, -1, 3)
    return mlp_output
