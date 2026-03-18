import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from PIL import Image
from scipy.stats import ortho_group
import os

def get_numpy_images(dataset: str, samples_per_class: int = 5000, max_samples_per_class: int = 5000, num_classes: int = 10):
    transform = transforms.ToTensor()

    if dataset == 'mnist':
        train_set = MNIST(root='/scratch/qingqu_root/qingqu1/alecx/mnist', train=True, download=True, transform=transform)

    elif dataset == 'fashion_mnist':
        train_set = FashionMNIST(root='/scratch/qingqu_root/qingqu1/alecx/fashion_mnist', train=True, download=True, transform=transform)

    elif dataset == 'cifar10':
        train_set = CIFAR10(root='/scratch/qingqu_root/qingqu1/alecx/cifar10', train=True, download=True, transform=transform)

    else:
        raise ValueError('Invalid dataset. Please choose from [mnist, fashion_mnist, cifar10]')

    images = []

    for k in range(num_classes):
        # Randomly sample images from current class
        images_k = np.array(train_set.data[np.array(train_set.targets) == k, ...])
        indices = np.random.randint(low=0, high=max_samples_per_class, size=(samples_per_class, ))
        images_k_subsample = images_k[indices, ...]

        # Reshape images 
        dim = np.prod(images_k_subsample.shape[1:])
        images.append(images_k_subsample.reshape((samples_per_class, dim)))

    return images

def main():
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    max_samples_per_class = [5000, 5000, 5000]
    num_classes = [10, 10, 10]

    for (i, dataset) in enumerate(datasets):
        print("===== DATASET:", dataset, " =====")
        images = get_numpy_images(dataset, 
                                samples_per_class=max_samples_per_class[i], 
                                max_samples_per_class=max_samples_per_class[i],
                                num_classes=num_classes[i])
        
        thresh = 0.95
        for k in range(num_classes[i]):
            images_k = images[k]
            U, svals, Vh = np.linalg.svd(images_k.T, full_matrices=False)
            fro_norm_ratios = np.sqrt(np.cumsum(svals**2)) / np.sqrt(np.sum(svals**2))

            print("Class " + str(k + 1) + " number of singular values =", np.argmax(fro_norm_ratios >= thresh))
        
        print("\n")
    


if __name__ == '__main__':
    main()