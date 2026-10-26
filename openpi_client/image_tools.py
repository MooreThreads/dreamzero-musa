from __future__ import annotations

import cv2
import numpy as np

# Pillow interpolation constants used by the official openpi-client API.
_PIL_TO_CV2_INTERPOLATION = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LANCZOS4,
    2: cv2.INTER_LINEAR,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_AREA,
    5: cv2.INTER_LINEAR,
}


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    method: int = 2,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize images with centered padding, matching openpi-client semantics."""
    grayscale = images.ndim == 2
    if grayscale:
        images = images[..., None]
    if images.ndim < 3:
        raise ValueError(f"Expected image shape [..., H, W, C], got {images.shape}")

    original_shape = images.shape
    flat_images = images.reshape(-1, *original_shape[-3:])
    interpolation = _PIL_TO_CV2_INTERPOLATION.get(method, cv2.INTER_LINEAR)
    resized = np.stack(
        [
            _resize_with_pad_cv2(image, height, width, interpolation, pad_value)
            for image in flat_images
        ]
    )
    output = resized.reshape(
        *original_shape[:-3],
        height,
        width,
        original_shape[-1],
    )
    return output[..., 0] if grayscale else output


def _resize_with_pad_cv2(
    image: np.ndarray,
    height: int,
    width: int,
    interpolation: int,
    pad_value: int,
) -> np.ndarray:
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    single_channel = image.shape[-1] == 1
    resize_input = image[..., 0] if single_channel else image
    resized = cv2.resize(
        resize_input,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    if single_channel:
        resized = resized[..., None]

    output_shape = (height, width, image.shape[-1])
    output = np.full(output_shape, pad_value, dtype=image.dtype)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    output[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
        :,
    ] = resized
    return output
