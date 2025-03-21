import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torchvision
from torchvision import transforms

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

def viz_images(ax, images, nrow=8):
    ax.imshow(make_grid(images, nrow=nrow).permute(1, 2, 0))
    ax.axis('off')
    return ax

def rand_partition(n=100):
    """
    - Generate a random partition of the interval [0, 1] into n intervals.
    - Uniformly samples from the simplex
    """
    cumulative_fracs = np.sort(np.concatenate([np.random.rand(n-1), [1]])) 
    cumulative_fracs[1:] = cumulative_fracs[1:] - cumulative_fracs[:-1]
    return cumulative_fracs

def load_tokenized_mnist(n_tokens, resolution, mode='uniform'):
    """
    - Load the MNIST dataset and discretize the pixel values into n_tokens.
    - mode: 'uniform' or 'quantile'. 
        - If 'uniform', the token values are uniformly spaced in the range of pixel values.
        - If 'quantile', the token values are chosen to be the quantiles of the pixel values.
    """
    if mode == 'quantile':
        raise NotImplementedError
    
    if os.path.exists(f'data/processed/MNIST/train_images_{n_tokens}_{resolution}x{resolution}.pt') == False:
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Resize((resolution, resolution))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        MAX_BATCH_SIZE = max(len(trainset), len(testset))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=MAX_BATCH_SIZE, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=MAX_BATCH_SIZE, shuffle=False)

        for i, data in enumerate(trainloader, 0):
            train_images, train_labels = data
            break

        for i, data in enumerate(testloader, 0):
            test_images, test_labels = data
            break

        data_min = train_images.min()
        data_max = train_images.max()

        train_images, token_vals = discretize_data( train_images, n_tokens, data_max=data_max, data_min=data_min)
        test_images, token_vals = discretize_data( test_images, n_tokens, data_max=data_max, data_min=data_min)
        os.makedirs('data/processed/MNIST', exist_ok=True)
        torch.save(train_images, f'data/processed/MNIST/train_images_{n_tokens}_{resolution}x{resolution}.pt')
        torch.save(train_labels, 'data/processed/MNIST/train_labels.pt')
        torch.save(test_images, f'data/processed/MNIST/test_images_{n_tokens}_{resolution}x{resolution}.pt')
        torch.save(test_labels, 'data/processed/MNIST/test_labels.pt')
        torch.save(token_vals, f'data/processed/MNIST/token_vals_{n_tokens}_{resolution}x{resolution}.pt')

    else:
        train_images = torch.load(f'data/processed/MNIST/train_images_{n_tokens}_{resolution}x{resolution}.pt')
        train_labels = torch.load('data/processed/MNIST/train_labels.pt')
        test_images = torch.load(f'data/processed/MNIST/test_images_{n_tokens}_{resolution}x{resolution}.pt')
        test_labels = torch.load('data/processed/MNIST/test_labels.pt')
        token_vals = torch.load(f'data/processed/MNIST/token_vals_{n_tokens}_{resolution}x{resolution}.pt')

    train_tokens = train_images.reshape(train_images.shape[0], -1)
    test_tokens = test_images.reshape(test_images.shape[0], -1)

    return train_tokens, train_labels, test_tokens, test_labels, token_vals

def tok2img(tokens, resolution, n_channels=1):
    return tokens.reshape(tokens.shape[0], n_channels, resolution, resolution)