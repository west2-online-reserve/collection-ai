import numpy as np
if __name__ == "__main__":
    grayscale_image=np.random.randint(0, 256, size=(200, 300))

    color_image = np.stack([grayscale_image, grayscale_image, grayscale_image], axis=-1)


    border_width = 20
    color_base = color_image.astype(np.float32)
    bordered_image = color_base.copy()


    w_left = np.linspace(0.0, 1.0, border_width, dtype=np.float32)[np.newaxis, :, np.newaxis]
    bordered_image[:, :border_width, :] = color_base[:, :border_width, :] * w_left

    w_right = np.linspace(0.0, 1.0, border_width, dtype=np.float32)[np.newaxis, :, np.newaxis]
    bordered_image[:, -border_width:, :] = color_base[:, -border_width:, :] * (1.0 - w_right) + 255.0 * w_right

    bordered_image = np.clip(bordered_image, 0, 255).astype(np.uint8)
    sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    sepia_image = color_image.astype(np.float32) @ sepia_matrix.T
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    alpha = 1.5
    color_f = sepia_image.astype(np.float32)
    luminance = (0.299 * color_f[..., 0] + 0.587 * color_f[..., 1] + 0.114 * color_f[..., 2])[..., np.newaxis]
    oversat_image = luminance + alpha * (color_f - luminance)
    oversat_image = np.clip(oversat_image, 0, 255).astype(np.uint8)
