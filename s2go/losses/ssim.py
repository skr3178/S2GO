"""SSIM — vendored from 3DGS reference implementation.

Source: kernel/gaussian-splatting/utils/loss_utils.py (Kerbl et al. 2023, INRIA).
License: 3DGS Inria research/non-commercial license. Pure-PyTorch path only;
the fusedssim CUDA path from the original is dropped (we don't have
diff_gaussian_rasterization compiled, and pure-PyTorch SSIM is fast enough
for our resolution).

Used by RGBRenderLoss (S2GO Eq. 8 third term, with form taken from 3DGS Eq. 7
since the S2GO paper does not specify L_rgb's formulation).
"""
from math import exp
import torch
import torch.nn.functional as F


def _gaussian(window_size, sigma):
    g = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)],
        dtype=torch.float32,
    )
    return g / g.sum()


def _create_window(window_size, channel):
    w_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    w_2d = w_1d.mm(w_1d.t()).unsqueeze(0).unsqueeze(0)
    return w_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         size_average: bool = True) -> torch.Tensor:
    """Structural Similarity Index between two batched images (channel-first).

    Args:
        img1, img2:   (N, C, H, W)  same shape
        window_size:  Gaussian window size (default 11, 3DGS standard)
        size_average: if True, returns scalar; else returns per-image tensor

    Returns:
        SSIM in [0, 1] (1 = identical). To use as a loss, take (1 - ssim).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"shape mismatch: {img1.shape} vs {img2.shape}")
    channel = img1.size(-3)
    window = _create_window(window_size, channel).type_as(img1)
    if img1.is_cuda:
        window = window.to(img1.device)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(-3, -2, -1))
