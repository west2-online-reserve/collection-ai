import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from work5.utils.util import GradCAM, show_cam_on_image, center_crop_img
from work5.model.vit import vit_base_patch16_224
# from torchvision import models

class ReshapeTransform:
    def __init__(self, model):
        # input_size = model.patch_embed.img_size
        # patch_size = model.patch_embed.patch_size
        input_size = [224, 224]
        patch_size = [16, 16]
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    model = vit_base_patch16_224()
    # model = models.vit_b_16()
    # torch.save(model.state_dict(), 'Caltech101_2.pth')
    model.load_state_dict(torch.load('Caltech101_1.pth'))
    model.eval()
    target_layers = [model.blocks[-1].norm1]
    # target_layers = [model.encoder.layers[-1].ln_1]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    # load image
    img_path = "D:\learn\Python\work5\image_0016.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 0  # accordion

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('GradCam_1')
    plt.show()


if __name__ == '__main__':
    main()
