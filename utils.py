import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def discretize_data(data, n_tokens, data_min=None, data_max=None):
    data_range = data_max - data_min
    data = ((data - data_min) / data_range) * (n_tokens - 1)
    data = data.long()
    token_vals = torch.arange(n_tokens) / (n_tokens - 1) * data_range + data_min
    return data, token_vals

def viz_masked_images(images, masks, nrow=8, color = (1,0,0)):

    MASKED_COLOR = torch.Tensor(color)
    images_perm = images.permute(0, 2, 3, 1)
    images_perm[masks.squeeze()] = MASKED_COLOR
    images = images_perm.permute(0, 3, 1, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(make_grid(images, nrow=nrow).permute(1, 2, 0))
    plt.axis('off')
    plt.show()