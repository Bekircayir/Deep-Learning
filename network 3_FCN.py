import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        # Netzstruktur definieren: 
        # 3 Fully-Connected Layer mit ReLU-Aktivierungsfunktionen dazwischen


    def forward(self, x):
        # Diese Funktion definiert den Forward-Pass

        return logits


if __name__ == "__main__":
    x = torch.rand((8, 2))
    model = NeuralNetwork()
    model.train()
    y = model(x)
    assert y.size() == (8, 2)
    torch.mean(y).backward()
    print("Sieht gut aus")
