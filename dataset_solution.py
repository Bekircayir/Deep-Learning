from sklearn.datasets import make_moons
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt


class Moons(Dataset):
    def __init__(self, size):
        super().__init__()
        self.x, self.y = make_moons(n_samples=size, noise=0.05)
        self.size = size

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return np.float32(self.x[index]), self.y[index]

    def plot(self):
        plt.scatter(self.x[:,0], self.x[:,1], c=self.y)
        plt.show()