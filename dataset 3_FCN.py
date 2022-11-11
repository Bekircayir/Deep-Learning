from sklearn.datasets import make_moons
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Moons(Dataset):
    def __init__(self, size):
        super().__init__()
        self.data = make_moons(n_samples=size, noise=0.01)
        self.size = size

    def get_mean(self):
        return np.mean(self.data[0], axis=0).astype(np.float32)

    def get_std(self):
        return np.std(self.data[0], axis=0).astype(np.float32)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[0][index].astype(np.float32), self.data[1][index]

    def plot(self):
        X = self.data[0]
        y = self.data[1]

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
        ax.set_title("Training data")
        plt.show()