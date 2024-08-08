#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

from matplotlib import pyplot as plt


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper



def get_expon_lr_func_with_restarts_and_decay(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, restart_steps=None, decay_factor=0.9, fine_tune_from=np.inf
):
    """
    Continuous learning rate decay function with warm restarts and decaying restarts.

    The learning rate is initially lr_init and decays to lr_final over max_steps.
    Optionally, restarts the learning rate schedule at given intervals or specific steps.
    Each restart reduces the initial learning rate by a decay factor.

    :param lr_init: initial learning rate
    :param lr_final: final learning rate
    :param lr_delay_steps: number of steps to delay learning rate decay
    :param lr_delay_mult: multiplier for initial learning rate delay
    :param max_steps: number of steps over which learning rate decays
    :param restart_steps: either an integer indicating interval steps or a list of specific steps to restart
    :param decay_factor: factor by which to decay the initial learning rate at each restart
    :return: function which takes step as input and returns learning rate
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0

        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        # if fine_tune_from is not None and step >= fine_tune_from:
        #     t = np.clip(step / max_steps, 0, 1)
        #     return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        if step < fine_tune_from:
            # Compute the number of completed cycles
            cycle_num = step // restart_steps
            # Compute step within the current cycle
            cycle_step = step % restart_steps
            # Decay the initial learning rate
            current_lr_init = lr_init * (decay_factor ** cycle_num)
            t = np.clip(cycle_step / max_steps, 0, 1)
        #
        else:
            # Compute the number of completed cycles
            cycle_num = fine_tune_from // restart_steps + 1
            # Compute step within the current cycle
            cycle_step = (step-fine_tune_from) % (max_steps - fine_tune_from)
            # Compute step within the current cycle
            # cycle_step = (step-fine_tune_from) % (max_steps)
            # Decay the initial learning rate
            current_lr_init = lr_init * (decay_factor ** cycle_num)
            t = np.clip(cycle_step / max_steps, 0, 1)


        log_lerp = np.exp(np.log(current_lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def plot_lr_schedule(lr_func, M):
    """
    Plot the learning rate over M iterations.

    :param lr_func: Learning rate function that takes an iteration step as input.
    :param M: Total number of iterations.
    """
    lr_values = [lr_func(step) for step in range(M)]

    plt.figure(figsize=(10, 6))
    plt.plot(range(M), lr_values)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.show()

def get_expon_lr_func_with_restarts_and_decay2(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, restart_steps=None, decay_factor=0.9, fine_tune_from=None
):
    """
    Continuous learning rate decay function with warm restarts and decaying restarts.

    The learning rate is initially lr_init and decays to lr_final over max_steps.
    Optionally, restarts the learning rate schedule at given intervals or specific steps.
    Each restart reduces the initial learning rate by a decay factor.
    After fine_tune_from steps, the learning rate converges smoothly without restarts.

    :param lr_init: initial learning rate
    :param lr_final: final learning rate
    :param lr_delay_steps: number of steps to delay learning rate decay
    :param lr_delay_mult: multiplier for initial learning rate delay
    :param max_steps: number of steps over which learning rate decays
    :param restart_steps: either an integer indicating interval steps or a list of specific steps to restart
    :param decay_factor: factor by which to decay the initial learning rate at each restart
    :param fine_tune_from: step from which no more restarts are performed
    :return: function which takes step as input and returns learning rate
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if fine_tune_from is not None and step >= fine_tune_from:
            t = np.clip(step / max_steps, 0, 1)
            return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        if isinstance(restart_steps, int):
            cycle_num = step // restart_steps
            cycle_step = step % restart_steps
            current_lr_init = lr_init * (decay_factor ** cycle_num)
            t = np.clip(cycle_step / restart_steps, 0, 1)
        elif isinstance(restart_steps, list):
            for i, restart in enumerate(restart_steps):
                if step < restart:
                    cycle_num = i
                    if i == 0:
                        cycle_step = step
                    else:
                        cycle_step = step - restart_steps[i-1]
                    break
            else:
                cycle_num = len(restart_steps)
                cycle_step = step - restart_steps[-1]

            current_lr_init = lr_init * (decay_factor ** cycle_num)
            if cycle_num == 0:
                t = np.clip(step / restart_steps[0], 0, 1)
            else:
                t = np.clip(cycle_step / (restart_steps[cycle_num] - restart_steps[cycle_num - 1]), 0, 1)
        else:
            current_lr_init = lr_init
            t = np.clip(step / max_steps, 0, 1)

        log_lerp = np.exp(np.log(current_lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_warm_restart_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, N=1000, alpha=0.9, beta=0.5):
    """
    Learning rate scheduler with warm restarts every N steps.
    The learning rate decays over N steps and then resets, with both the initial
    and final learning rates decaying by factors alpha and beta respectively.

    :param lr_init: initial learning rate
    :param lr_final: final learning rate
    :param lr_delay_steps: steps for delayed start of learning rate decay
    :param lr_delay_mult: multiplier for learning rate delay
    :param N: number of steps between warm restarts
    :param alpha: decay factor for initial learning rate after each restart
    :param beta: decay factor for final learning rate after each restart
    :return: function that takes step as input and returns the learning rate
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        cycle = step // N
        t = (step % N) / N

        lr_init_t = lr_init * (alpha ** cycle)
        lr_final_t = lr_final * (beta ** cycle)

        log_lerp = np.exp(np.log(lr_init_t) * (1 - t) + np.log(lr_final_t) * t)

        return delay_rate * log_lerp

    return helper
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
