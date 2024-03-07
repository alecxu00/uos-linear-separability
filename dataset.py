import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.stats import ortho_group

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

def uos_dataset(N_k, K, d, r, orthogonal=False, batch_size=128, seed=0, angle=0):
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
        assert d > (K * r) # Assert data dimension is large enough
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

    # Randomly shuffle samples and labels
    perm = np.random.permutation(N)
    samples = samples[perm, :]
    labels = labels[perm]

    # Create data loader
    mog_dataset = MoGDataset(samples, labels)
    mog_loader = DataLoader(mog_dataset, batch_size=batch_size, shuffle=True)

    return mog_dataset, mog_loader
