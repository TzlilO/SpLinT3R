import wandb
import torch
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def logger_function(epoch, reconstructed_image, gt_image, loss, step):
    # Convert tensors to numpy arrays
    reconstructed_np = reconstructed_image.detach().cpu().numpy()
    gt_np = gt_image.detach().cpu().numpy()

    # Ensure images are in the correct shape (H, W, C)
    if reconstructed_np.shape[0] == 3:
        reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))
    if gt_np.shape[0] == 3:
        gt_np = np.transpose(gt_np, (1, 2, 0))

    # Normalize images to [0, 1] range if needed
    reconstructed_np = (reconstructed_np - reconstructed_np.min()) / (reconstructed_np.max() - reconstructed_np.min())
    gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min())

    # Calculate PSNR and SSIM
    psnr_value = psnr(gt_np, reconstructed_np, data_range=1)
    ssim_value = ssim(gt_np, reconstructed_np, multichannel=True, data_range=1)

    # Create a grid of images
    img_grid = torchvision.utils.make_grid([
        torch.from_numpy(gt_np.transpose(2, 0, 1)),
        torch.from_numpy(reconstructed_np.transpose(2, 0, 1))
    ])

    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "images": wandb.Image(img_grid, caption="GT vs Reconstructed")
    })
