import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import USPS
from PIL import Image
from scipy.stats import ortho_group
import os

# Dataset of synthetic samples
class SyntheticDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return self.labels.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx]

        return sample, label

def get_uos_dataset(N_k, K, d, r, orthogonal=False, batch_size=128, seed=0, angle=0):
    '''
    N_k: number of samples per class
    K: number of classes
    d: data dimension
    r: subspace ranks
    orthogonal: make subspace bases orthogonal to each other
    batch_size: batch size
    angle: minimum principal angle (in degrees) between pairs of subspaces
        - angle = 0 --> generate subspaces uniformly at random
    '''

    N = N_k * K # total number of samples

    # Generate subspace bases randomly
    np.random.seed(seed)

    if angle == 0: # Generate subspaces randomly
        if orthogonal:
            U = ortho_group.rvs(d)
            bases = [ U[:, i*r:(i+1)*r] for i in range(K) ]
        else:
            bases = [ortho_group.rvs(d)[:, :r] for _ in range(K)]

    else: # Generate subspaces such that each pair of bases has a minimum (and maximum) principal angle of angle
        assert K % 2 == 0 # Even number of subspaces
        assert d >= (K * r) # Assert data dimension is large enough
        angle_rad = np.deg2rad(angle) # Convert degrees to radians

        U_full = ortho_group.rvs(d)
        bases = []
        for k in range(K//2):
            U1 = np.zeros((d, r)) # First subspace in pair
            U2 = np.zeros((d, r)) # Second subspace
            for rr in range(r):
                idx_start = rr*2 + 2*k*r
                idx_end = (rr+1)*2 + 2*k*r
                u = U_full[:, idx_start:idx_end]
                U1[:, rr] = u[:, 0] # Basis vector for first subspace
                U2[:, rr] = np.cos(angle_rad)*u[:, 0] + np.sin(angle_rad)*u[:, 1] # Basis vector for second subspace

            bases.append(U1)
            bases.append(U2)


    # Create samples and labels
    samples = np.array( [bases[i] @ np.random.randn(r, N_k) for i in range(K)] ) # Randomly generate samples in each subspace
    samples = np.transpose(samples, axes=(0, 2, 1)) # N_k x K x d
    samples = np.float32( np.reshape(samples, (N, d)) ) # N x d

    labels = np.arange(K)
    labels = np.repeat(labels, N_k, axis=0)

    # Randomly shuffle samples and labels
    perm = np.random.permutation(N)
    samples = samples[perm, :]
    labels = labels[perm]

    # Create data loader
    uos_dataset = SyntheticDataset(samples, labels)
    uos_loader = DataLoader(uos_dataset, batch_size=batch_size, shuffle=True)

    return uos_dataset, uos_loader


def get_mog_dataset(N_k, K, d, batch_size=128, seed=0):
    '''
    N_k: number of samples per class
    K: number of classes
    d: data dimension
    batch_size: batch size
    '''

    N = N_k * K # total number of samples

    # Generate Gaussian parameters randomly
    np.random.seed(seed)
    means = [2. * np.random.rand(d) - 1. for _ in range(K)] # Mean vectors
    tmp = [2. * np.random.rand(d, d) - 1. for _ in range(K)]
    covs = [tmp[i].T @ tmp[i] for i in range(K)] # Covariance matrices

    # Create samples and labels
    samples = np.array( [np.random.multivariate_normal(means[i], covs[i], size=N_k) for i in range(K)] ) # Randomly sample from Multivariate Gaussian
    samples = np.transpose(samples, axes=(0, 2, 1)) # N_k x K x d
    samples = np.float32( np.reshape(samples, (N, d)) ) # N x d

    labels = np.arange(K)
    labels = np.repeat(labels, N_k, axis=0)

    # Randomly shuffle samples and labels
    perm = np.random.permutation(N)
    samples = samples[perm, :]
    labels = labels[perm]

    # Create data loader
    mog_dataset = SyntheticDataset(samples, labels)
    mog_loader = DataLoader(mog_dataset, batch_size=batch_size, shuffle=True)

    return mog_dataset, mog_loader


# Partial dataset for USPS digits
class USPSPartialDataset(USPS):
    def __init__(self, root, N_k, K, train=True, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)

        assert K > 0 and K < 11, "Invalid number of USPS classes."
        self.num_classes = K # Number of classes
        self.num_samples = N_k * K # Number of samples

        self.targets = np.array(self.targets)
        all_samples = []
        all_labels = []
        for k in range(K):
            samples_in_class = self.data[self.targets == k, :][:N_k]
            all_samples.append(samples_in_class)
            all_labels.append( np.ones(len(samples_in_class)) * k )

        self.samples = np.concatenate(all_samples, 0)
        self.labels = np.concatenate(all_labels, 0)

    def __getitem__(self, idx):
        img = self.samples[idx]
        img = Image.fromarray(img)
        #img = img.resize((img.size[0] // 2, img.size[1] // 2))
        img = np.array(img, dtype=np.float32)

        label = self.labels[idx]
        return img.flatten(), label

    def __len__(self):
        return len(self.samples)


def get_usps_dataset(N_k, K, batch_size=128, root='./datasets/', train=True):
    '''
        N_k: number of samples per class
        K: number of classes 
        batch_size: batch size
        root: directory to store downloaded images
        train: indicate train or test set
    '''

    usps_dataset = USPSPartialDataset(root, N_k, K, train=train)
    usps_loader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True)

    return usps_dataset, usps_loader


# Partial dataset for MCR2 features of CIFAR-10
class CIFAR10MCR2PartialDataset(Dataset):
    def __init__(self, N_k, K, train=True, root='./datasets/cifar-10/', features_fname='train_features.npy', labels_fname='train_labels.npy'):

        assert K > 0 and K < 11, "Invalid number of CIFAR-10 classes."

        features_path = os.path.join(root, features_fname)
        labels_path = os.path.join(root, labels_fname)

        features = np.load(features_path)
        labels = np.load(labels_path)

        assert N_k <= features.shape[0] // 10, "Invalid number of samples per class."

        all_samples = []
        all_labels = []
        for k in range(K):
            samples_in_class = features[labels == k, :][:N_k]

            all_samples.append(samples_in_class)
            all_labels.append( np.ones(samples_in_class.shape[0]) * k )

        self.samples = np.concatenate(all_samples, 0).astype(np.float32)
        self.labels = np.concatenate(all_labels, 0)

    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx]

        return sample, label

    def __len__(self):
        return len(self.labels)

def get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10/', features_fname='train_features.npy', labels_fname='train_labels.npy', batch_size=128):
    '''
        - N_k: number of samples per class
        - K: number of classes
        - root: directory where dataset is stored
        - features_fname: filename of features
        - labels_fname: filename of labels
        - batch_size: batch size
    '''

    cifar10_mcr2_dataset = CIFAR10MCR2PartialDataset(N_k, K, root=root, features_fname=features_fname, labels_fname=labels_fname)
    cifar10_mcr2_loader = DataLoader(cifar10_mcr2_dataset, batch_size=batch_size, shuffle=True)

    return cifar10_mcr2_dataset, cifar10_mcr2_loader
