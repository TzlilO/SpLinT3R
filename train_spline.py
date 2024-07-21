#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

import os

import imageio
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim, pw_cosine_similarity
from gaussian_renderer import render, network_gui, render_spline
from scene import Scene, SplineScene
from utils.general_utils import safe_state
import uuid
import sys
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.data import *
from model import SplineModel
torch.autograd.set_detect_anomaly(True)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def render_view(images):
    # if not viewpoint_stack:
    writer = imageio.get_writer(os.path.join("experiments/tpot", f'view_evolve_2.mp4'), fps=10)

    for image in images:
        # Add images to the video writer
        writer.append_data(tensor_to_numpy(image))

    writer.close()
def tensor_to_numpy(image_tensor):
    # Assuming image_tensor is of shape (C, H, W) and values are in the range [0, 1]
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C) and move to CPU
    image_np = np.clip((image_np * 255),a_min=0, a_max=255).astype(np.uint8) # Convert to uint8

    return image_np
def render_scene(scene, gaussians, pipe, bg, iteration=None):
    # if not viewpoint_stack:
    writer = imageio.get_writer(os.path.join("experiments/tpot3", f'scene2_360_after_{iteration}_iter.mp4'), fps=4)

    viewpoint_stack = scene.getTrainCameras().copy()

    while len(viewpoint_stack) - 1 > 0:
        viewpoint_cam = viewpoint_stack.pop()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        # Add images to the video writer
        writer.append_data(tensor_to_numpy(image))

    writer.close()


def spline_splatting_training(dataset, opt, pipe, testing_iterations, checkpoint_iterations, debug_from):
    first_iter = 0
    device = 'cuda'

    # Load Spline reference scene
    file_path = 'experiments/bpt-data/teapot.txt'
    file_content = read_bpt_file(file_path)
    patches = parse_bpt(file_content)
    splines = SplineModel(patches, device=device, resolution=16)

    tb_writer = prepare_output_and_logger(dataset)
    scene = SplineScene(dataset, splines)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    step_interval = splines.step_size
    splines.sample_gaussians()
    loss = torch.tensor([0], dtype=torch.float32, device="cuda")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        if iteration in [2500, 5000, 8500, 10000, 12500]:
            splines.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render_spline(viewpoint_cam, splines, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        # L_geo = pw_cosine_similarity(splines.surface_normals, splines.areas, weight=0.1)
        ssim_term = (1.0 - ssim(image, gt_image))
        loss += ((1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term)

        if not iteration % step_interval:
            loss.backward()
            splines.step(iteration, visibility_filter)
            loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            if iteration % splines.splitting_interval_every == 0 and  iteration <= 6000:
                splines.patch_upsampler()
            splines.sample_gaussians()
        iter_end.record()

        with torch.no_grad():
            if iteration % 500 == 1:
                compare_gt2splat(gt_image, image, viewpoint_cam)

            # splines.scale_shaking(iteration)
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((splines.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            spline_training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, splines, render_spline, (pipe, background))


def compare_gt2splat(gt_image, image, viewpoint_cam):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    ax[0].imshow(gt_image.cpu().detach().permute(1, 2, 0).numpy())
    ax[0].set_title(f"Ground Truth image ID: {viewpoint_cam.uid}")
    ax[0].axis('off')
    ax[1].imshow(image.clone().cpu().detach().permute(1, 2, 0).numpy())
    ax[1].set_title(f"Splatted image ID: {viewpoint_cam.uid}")
    ax[1].axis('off')
    plt.axis('off')
    plt.show()
    plt.close()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def spline_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, splines, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': [scene.getTestCameras()[1]]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, splines, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: SSIM {}, L1 {} PSNR {}".format(iteration, config['name'], ssim_test, l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", splines.get_opacity, iteration)
            tb_writer.add_scalar('total_points', splines.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(1, 31)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 9_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    spline_splatting_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.checkpoint_iterations, args.debug_from)
    # All done
    print("\nTraining complete.")


def parse_config_args():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1_000, 2_000, 4_000, 7_000, 9_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 9_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    return args, lp, op, pp


if __name__ == '__main__':
    # General Idea for the experiment
    main()
