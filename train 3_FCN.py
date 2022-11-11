from network import NeuralNetwork
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataset import Moons

# Größe des Datensatzes
num_train = 1024
num_val = 128

# Trainingsparameter
num_epochs = 100
learning_rate = 0.001
batch_size = 128

# Datensatz laden und Dataloader initialisieren
train_data = Moons(num_train)
val_data = Moons(num_val)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=num_val, shuffle=False)

# Mittelwert und Standardabweichung vom Trainingsset zur Normalisierung
train_mean = torch.from_numpy(train_data.get_mean()).unsqueeze(0)
train_std = torch.from_numpy(train_data.get_std()).unsqueeze(0)

# Trainingsdaten anzeigen
train_data.plot()

# Neuronales Netz, Loss-Funktion, Optimizer definieren

# Aufgabe 2.1 a)
model = 
# Aufgabe 2.1 b)
loss_function =
# Aufgabe 2.1 c)
optimizer = 

train_loss = []
val_loss = []
val_accuracies = []

iteration = 0
for epoch in range(num_epochs):
    print("epoch %d / %d" % (epoch+1, num_epochs))

    # Aufgabe 2.2 a)
    # Neuronales Netz in Trainingsmodus versetzen


    # Batch-weise über den Trainingsdatensatz iterieren
    for batch_idx, (x, y) in enumerate(train_dataloader):
        iteration += 1

        # Gradienten zurücksetzen
        optimizer.zero_grad()

        # Aufgabe 2.2 b)
        # Daten normalisieren
        x = 

        # Aufgabe 2.2 c)
        # Forward-Pass
        y_estimate = 

        # Aufgabe 2.2 d)
        # Loss berechnen
        loss = 


        train_loss += [(iteration, loss.item())]
        print("%4d / %4d -- loss: %.3f" % (batch_idx+1, len(train_dataloader), loss.item()))

        # Backward-Pass (Gradienten berechnen)
        loss.backward()

        # Ein Schritt des Gradientenabstiegs mit dem Optimizer
        optimizer.step()
    
    # Aufgabe 2.3 a)
    # Neuronales Netz in Evaluationsmodus versetzen
    

    # Accuracy auf dem Validationdatensatz berechnen
    x, y = next(iter(val_dataloader))

    # Aufgabe 2.3 b)
    # Daten normalisieren
    x = 

    # Aufgabe 2.3 c)
    # Forward-Pass
    y_estimate = 


    # Aufgabe 2.3 d)
    # IDs bestimmen
    y_estimate_ids = 


    # Aufgabe 2.3 e)
    # Accuracy bestimmen
    accuracy = 


    val_accuracies += [(iteration, accuracy)]
    print("val accuracy: %.3f %%" % (accuracy * 100))
    
# Trainingsloss und Validation-Accuracy über die Trainingsiterationen plotten
train_loss = np.array(train_loss)
val_accuracy = np.array(val_accuracies)
    
plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(train_loss[:, 0], train_loss[:, 1])
ax.set_title("Training Loss")
plt.show()

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(val_accuracy[:, 0], val_accuracy[:, 1])
ax.set_title("Validation Accuracy")
plt.show()

