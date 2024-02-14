import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.sparse.linalg import svds

# Dataset of samples that lie on union of subspaces
class UoSDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return self.labels.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx]

        return sample, label

def uos_dataset(N_k, K, d, r, orthogonal=False, batch_size=128, seed=0):
    '''
    N_k: number of samples per class
    K: number of classes
    d: data dimension
    r: subspace ranks
    orthogonal: make subspace bases orthogonal to each other
    batch_size: batch size
    '''

    N = N_k * K # total number of samples

    # Generate subspace bases randomly
    np.random.seed(seed)
    if orthogonal:
        U, _, _ = np.linalg.svd(np.random.rand(d, d))
        bases = [ U[:, i*r:(i+1)*r] for i in range(K) ]
    else:
        bases = [svds(np.random.rand(d, d), r)[0] for _ in range(K)]
    
    # Create samples and labels
    samples = np.array( [bases[i] @ (2*np.sqrt(3)*np.random.rand(r, N_k)- np.sqrt(3)) for i in range(K)] ) # Randomly generate samples in each subspace
    samples = np.transpose(samples, axes=(0, 2, 1)) # N_k x K x d
    samples = np.float32( np.reshape(samples, (N, d)) ) # N x d

    labels = np.arange(K)
    labels = np.repeat(labels, N_k, axis=0)

    # Create data loader
    uos_dataset = UoSDataset(samples, labels)
    uos_loader = DataLoader(uos_dataset, batch_size=batch_size, shuffle=True)

    return uos_dataset, uos_loader



# Dataset of samples that come from mixture of (high-rank) Gaussians
class MoGDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return self.labels.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx]

        return sample, label

def mog_dataset(N_k, K, d, batch_size=128, seed=0):
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

    # Create data loader
    mog_dataset = MoGDataset(samples, labels)
    mog_loader = DataLoader(mog_dataset, batch_size=batch_size, shuffle=True)

    return mog_dataset, mog_loader
