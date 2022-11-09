import torch
import matplotlib.pyplot as plt

epochs = 20
learning_rate = 0.1

use_gpu = False

if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
    # alternativ: device=None verwendet automatisch die CPU
    device = None

# Datensatz-Klasse definieren:
class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_size=100) -> None:
        super().__init__()
        self.dataset_size = dataset_size
        self.x = torch.linspace(0, 10, dataset_size, dtype=torch.float32, device=device)
        self.y = 2*self.x + 1
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# Aufgabe a)
# Dataset-Objekt erstellen
dataset = ...

# Aufgabe b)
# Dataloader-Objekt erstellen
dataloader = ...

weights = torch.rand(2, dtype=torch.float32, device=device, requires_grad=True) 
# Aufgabe c)
# Optimizer-Objekt erstellen
optimizer = ...

# Aufgabe d)
# For-Schleifen für das Training



# Daten und optimierte Gerade plotten:
plt.figure()
x = dataset.x
y = dataset.y
X = torch.stack([x, torch.ones_like(x)], dim=-1)
plt.plot(x.numpy(), y.numpy(), 'kx')
y_final = torch.sum(X*weights, dim=-1)
plt.plot(x.numpy(), y_final.detach().numpy(), 'g-')
plt.show()

    