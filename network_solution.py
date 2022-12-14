from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Netzstruktur definieren: 
        # 3 Fully-Connected Layer mit ReLU-Aktivierungsfunktionen dazwischen
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax()
        )

    def forward(self, x):
        # Diese Funktion definiert den Forward-Pass
        return self.linear_relu_stack(x)

