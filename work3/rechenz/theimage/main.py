import numpy as np

grayscale_image = np.random.randint(0, 256, size=(200, 300))
color_image = np.stack([grayscale_image]*3, axis=-1)
print(color_image.shape)
sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
])
ans_image = np.dot(color_image, sepia_matrix.T)
ans_image = np.clip(ans_image, 0, 255).astype(np.uint8)  # AI真NB
print(ans_image)
Luminance = np.dot(color_image, [0.299, 0.587, 0.114]).astype(np.uint8)
Luminance = np.repeat(np.expand_dims(Luminance, axis=-1), 3, axis=-1)
alpha = 1.5
new_image = np.clip(Luminance+alpha*(ans_image-Luminance),
                    0, 255).astype(np.uint8)
print(new_image)
left_image = np.linspace(0, color_image[0][0], 20, dtype='uint8')
left_image = np.repeat(np.expand_dims(left_image, axis=0), 200, axis=0)
print(left_image.shape)
right_image = np.linspace(color_image[0][0], 0, 20, dtype='uint8')
right_image = np.repeat(np.expand_dims(right_image, axis=0), 200, axis=0)
print(right_image.shape)
ansss_image = np.concatenate([left_image, color_image, right_image], axis=1)
print(ansss_image.shape)
