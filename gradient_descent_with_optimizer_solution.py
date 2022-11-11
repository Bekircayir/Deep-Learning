import torch
import matplotlib.pyplot as plt
import numpy as np

iterations = 100
learning_rate = 0.1
# Frage: was passiert wenn wir die Lernrate zu groß/klein wählen?

use_gpu = False

if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
    # alternativ: device=None verwendet automatisch die CPU
    device = None

# Daten definieren:
x = torch.linspace(0, 10, 10, dtype=torch.float32, device=device)
y = 2*x+1
# Jeden Wert in x zu einem Vektor X=[x, 1] erweitern, 
# um die Berechnung von ax+b zu vereinfachen:
X = torch.stack([x, torch.ones_like(x)], dim=-1)

# Die Daten beschreiben offensichtlich die Gerade y = 2*x+1 (siehe oben)
# Wir tun nun so, als wuessten wir das nicht, und wollen via Gradientenabstieg 
# mithilfe von PyTorch die Parameter a,b der Geraden ax+b bestimmen.

# Parameter zufällig initialisieren:
# Im Kontext von Neuronalen Netzen oft auch als Gewichte (weights) bezeichnet
# Zwei zufällige Zahlen aus der Gleichverteilung [0,1]
weights = torch.rand(2, dtype=torch.float32, device=device, requires_grad=True) 

# Aufgabe 2a)
optimizer = torch.optim.Adam(params=[weights], lr=learning_rate)

for i in range(iterations):

    # Aufgabe 2a)
    # Gradienten im Optimizer zurücksetzen:
    optimizer.zero_grad()

    # X hat die Dimension 10x2
    # weights hat Dimension 2
    # Für die Berchnung X*weights müssen wir daher noch eine Dimension zu 
    # weights hinzufuegen:
    w = weights.view(1, 2)  # analog zu numpy.reshape

    # Aufgabe 2b)
    # y=ax+b berechnen:
    y_estimate = torch.sum(X * w, dim=-1)

    # Aufgabe 2c)
    # Fehler ("Loss") bezueglich der bekannten y-Werte berechnen:
    loss = torch.mean((y-y_estimate)**2)

    print("Iteration %d - Loss: %.4f" % (i, loss.item()))

    # Aufgabe 2d)
    # mit .backward() werden die Gradienten d loss / d weights berechnet
    loss.backward()

    # Aufgabe 2e)
    optimizer.step()
    
# Daten und optimierte Gerade plotten:
plt.figure()
plt.plot(x.numpy(), y.numpy(), 'kx')
y_final = torch.sum(X*weights, dim=-1)
plt.plot(x.numpy(), y_final.detach().numpy(), 'g-')
plt.show()

    