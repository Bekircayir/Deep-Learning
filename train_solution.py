import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np

# Importiere Klassen
from dataset import Moons
from network import NeuralNetwork

# Größe des Datensatzes
num_train = 1024
num_val = 128

# Trainingsparameter
num_epochs = 500
learning_rate = 0.1
batch_size = 128

# Datensatz laden und Dataloader initialisieren
train_data = Moons(num_train)
val_data = Moons(num_val)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Trainingsdaten anzeigen
#train_data.plot()

# Neuronales Netz, Loss-Funktion, Optimizer definieren
model = NeuralNetwork()

# Test für die Klassen
x, y = next(iter(train_dataloader))
print(model(x))

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

iteration = 0
for epoch in range(num_epochs):
    print("epoch %d / %d" % (epoch+1, num_epochs))

    # Neuronales Netz in Trainingsmodus versetzen
    model.train()

    # Batch-weise über den Trainingsdatensatz iterieren
    for batch_idx, (X, y) in enumerate(train_dataloader):
        iteration += 1

        # Gradienten zurücksetzen
        optimizer.zero_grad()

        # Forward-Pass
        pred = model(X)

        # Loss berechnen
        loss = loss_function(pred, y)
        #print("Loss: ", loss.item())

        # Backward-Pass (Gradienten berechnen)
        loss.backward()

        # Ein Schritt des Gradientenabstiegs mit dem Optimizer
        optimizer.step()
    
    # Neuronales Netz in Inferenzmodus versetzen
    model.eval()

    # Accuracy auf dem Validationdatensatz berechnen
    x_test, y_test = next(iter(val_dataloader))
    pred_test = model(x_test)
    pred_test = torch.argmax(pred_test, dim=1)
    accuracy = (torch.sum(pred_test == y_test)/batch_size).item()
    print("Genauigkeit: ", accuracy)

    


