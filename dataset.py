import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
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


# Partial dataset for CIFAR-10
class CIFAR10PartialDataset(CIFAR10):
    def __init__(self, root, K, N_k, train=True, transform=None, download=True, **kwargs):
        super().__init__(root=root, train=train, transform=transform, download=download, **kwargs)

        if K == 0:
            self.data = []
            self.targets = []
        else:
            print(f"Using Cifar10 with only {K} classes! \n")
            self.targets = np.array(self.targets)
            all_data = []
            all_label = []
            for k in range(K):
                idx = np.random.randint(low=0, high=np.count_nonzero(self.targets == k), size=(N_k,))
                all_data.append(self.data[self.targets == k][idx])
                all_label.append(np.ones(N_k) * k)

            self.data = np.concatenate(all_data, 0)
            self.targets = np.concatenate(all_label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img.flatten(), target


def get_cifar10_dataset(N_k, K, data_dir, batch_size, train=True, shuffle=True):
    """
    args:
    @ data_dir: where dataset are stored or to be stored
    @ batch_size: training and testing batch size
    @ K: how many classes to use for the dataset
    """

    # Transform
    transform_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


    data_set = CIFAR10PartialDataset
    transform_data = transform_cifar10

    cifar10_dataset = data_set(data_dir, K, N_k,
                        train=train, transform=transform_data)

    # Dataloader
    cifar10_loader = torch.utils.data.DataLoader(
            cifar10_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return cifar10_dataset, cifar10_loader


# Partial dataset for MCR2 features of CIFAR-10
class CIFAR10MCR2PartialDataset(Dataset):
    def __init__(self, N_k, K, train=True, root='./datasets/cifar10_mcr2/', features_fname='train_features.npy', labels_fname='train_labels.npy'):

        assert K > 0 and K < 11, "Invalid number of CIFAR-10 classes."

        features_path = os.path.join(root, features_fname)
        labels_path = os.path.join(root, labels_fname)

        features = np.load(features_path)
        labels = np.load(labels_path)

        assert N_k <= features.shape[0] // 10, "Invalid number of samples per class."

        all_samples = []
        all_labels = []
        for k in range(K):
            idx = np.random.randint(low=0, high=np.count_nonzero(labels == k), size=(N_k,))
            samples_in_class = features[labels == k, :][idx]

            all_samples.append(samples_in_class)
            all_labels.append( np.ones(N_k) * k )

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
