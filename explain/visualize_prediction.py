
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Union

from explain.importance_scores import get_importance
from lvq.model import Model


def region_importance_image(model: Model, sample: Tensor) -> Tensor:
    """Perform partial forward pass to compute region effect map."""
    with torch.no_grad():
        feature, subspace, Vh, S, output = model.forward_partial(sample)
        region_effect_map, region_effect_map_plus, _ = get_importance(
            model, feature, Vh, S, output,
            model.prototype_layer.xprotos,
            model.prototype_layer.relevances
        )
    # return region_effect_map
    return region_effect_map_plus.sum(dim=1).squeeze()


def create_heatmap(
        effect_map: Union[Tensor, np.array],
        output_path: str,
        img_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Generate and save heatmap from effect map."""

    if isinstance(effect_map, Tensor):
        effect_map_np = effect_map.cpu().numpy()
    else:
        effect_map_np = effect_map
    rescaled_map = effect_map_np - np.amin(effect_map_np)
    rescaled_map = rescaled_map / (np.amax(rescaled_map) + 1e-8)

    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_map), cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap


def overlay_heatmap_on_image(
        heatmap: np.ndarray,
        sample_array: np.ndarray,
        output_path: str
):
    """Overlay heatmap on original image and save."""

    overlay = 0.6 * sample_array / 255 + 0.2 * heatmap
    plt.imsave(fname=output_path, arr=overlay, vmin=0.0, vmax=1.0)


def save_heatmaps(
        sample_array: np.ndarray,
        region_effect_map: Tensor,
        result_dir: str
):
    """Generate and save heatmaps for region effect maps."""

    os.makedirs(result_dir, exist_ok=True)

    # Save latent heatmap
    latent_heatmap_path = os.path.join(result_dir, 'heatmap_latent_map.png')
    _ = create_heatmap(region_effect_map, latent_heatmap_path)

    # Resize to image size and save the (upsampled) heatmap
    upsampled_effect_map = cv2.resize(
        region_effect_map.numpy(),
        dsize=(sample_array.shape[1], sample_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    upsampled_heatmap_path = os.path.join(result_dir, 'heatmap_upsampled.png')
    upsampled_heatmap = create_heatmap(upsampled_effect_map, upsampled_heatmap_path)

    overlay_path = os.path.join(result_dir, 'heatmap_original_image.png')
    overlay_heatmap_on_image(upsampled_heatmap, sample_array, overlay_path)


def compute_pixel_importance(
    sample_array: np.ndarray,
    region_importances: Tensor,
    result_dir: str
):
    """Compute and save heatmaps for pixel-level importance."""
    save_heatmaps(sample_array, region_importances[0], result_dir)


def compute_region_importance(model: Model, imgs_tensor: Tensor) -> Tensor:
    """Compute region importance for all images in batch."""

    region_effect_maps = [
        region_importance_image(model, sample.unsqueeze(0))
        for sample in imgs_tensor
    ]

    return torch.stack(region_effect_maps, dim=0)

