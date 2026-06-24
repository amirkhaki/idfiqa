"""Visualization utilities for spatial importance maps."""
import torch
import numpy as np


def overlay_sensitivity(ref_image, sensitivity_map, output_path,
                        alpha=0.5, cmap="jet"):
    """
    Overlay a sensitivity map on a reference image.

    Args:
        ref_image: (C, H, W) tensor or (H, W, C) numpy array
        sensitivity_map: (grid_h, grid_w) tensor
        output_path: path to save the figure
        alpha: overlay opacity
        cmap: matplotlib colormap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(ref_image, torch.Tensor):
        ref_image = ref_image.permute(1, 2, 0).cpu().numpy()
    if ref_image.max() <= 1.0:
        ref_image = (ref_image * 255).astype(np.uint8)

    if isinstance(sensitivity_map, torch.Tensor):
        sensitivity_map = sensitivity_map.cpu().numpy()

    H, W = ref_image.shape[:2]
    grid_h, grid_w = sensitivity_map.shape

    upsampled = np.kron(sensitivity_map, np.ones((H // grid_h, W // grid_w)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ref_image)
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(ref_image)
    im = axes[1].imshow(upsampled, cmap=cmap, alpha=alpha,
                         extent=[0, W, H, 0], vmin=0, vmax=sensitivity_map.max())
    axes[1].set_title("Spatial Importance")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
