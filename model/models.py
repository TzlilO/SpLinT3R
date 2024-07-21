import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlPointsRefinement(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, device='cuda'):
        super(ControlPointsRefinement, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim, device=device), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            layers.append(nn.ReLU().to(device))
        layers.append(nn.Linear(hidden_dim, output_dim, device=device))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)