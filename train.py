from network import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os

# Trainingsparameter
num_epochs = 10
learning_rate = 0.001
batch_size = 64

# Tranformationen zur Vorverarbeitung definieren:
# Mittelwert und Standardabweichung der Trainingsdaten (zur Normalisierung):
data_mean_and_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Aufgabe 2
# Augmentierung der Daten während des Trainings:
train_transforms = transforms.Compose([
transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
transforms.RandomHorizontalFlip(),
# Konvertierung von Numpy-Array zu Pytorch-Tensor:
transforms.ToTensor(),
# Normalisierung:
transforms.Normalize(*data_mean_and_std)
])
# Während Inferenz auf dem Validation-Datensatz nur Normalisierung (keine Augmentierung):
val_transforms = transforms.Compose([
# Konvertierung von Numpy-Array zu Pytorch-Tensor:
transforms.ToTensor(),
# Normalisierung:
transforms.Normalize(*data_mean_and_std)
])

# CIFAR10-Datensatz laden. Die oben definierten Transformationen werden automatisch angewendet:
train_data = CIFAR10("cifar10", train=True, transform=train_transforms, download=True)
val_data = CIFAR10("cifar10", train=False, transform=val_transforms, download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Neuronales Netz, Loss-Funktion, Optimizer definieren
model = SimpleCNN()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
val_loss = []
val_accuracies = []

os.makedirs("./checkpoints", exist_ok=True)

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

        train_loss += [(iteration, loss.item())]
        print("batch %4d / %4d -- loss: %.3f Heir fonkioniert oder? " % (batch_idx+1, len(train_dataloader), loss.item()), end="\r")

        # Backward-Pass (Gradienten berechnen)
        loss.backward()

        # Ein Schritt des Gradientenabstiegs mit dem Optimizer
        optimizer.step()
    
    print("")

    # Neuronales Netz in Inferenzmodus versetzen
    model.eval()

    # Accuracy auf dem Validationdatensatz berechnen
    val_accuracies_epoch = []
    for batch_idx, (X, y) in enumerate(val_dataloader):
        pred = model(X)
        pred = torch.argmax(pred, dim=-1)
        accuracy = torch.mean((pred == y).float()).item()
        val_accuracies_epoch += [accuracy]
    val_accuracy = np.mean(val_accuracies_epoch)
    val_accuracies += [(iteration, val_accuracy)]
    print("epoch %d --val accuracy: %.2f\n *** " % (epoch+1, val_accuracy * 100))
    
# Trainingsloss und Validation-Accuracy über die Trainingsiterationen plotten
train_loss = np.array(train_loss)
val_accuracy = np.array(val_accuracies)
    
plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(train_loss[:, 0], train_loss[:, 1])
ax.set_title("Training Loss")

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(val_accuracy[:, 0], val_accuracy[:, 1])
ax.set_title("Validation Accuracy")
plt.show()

