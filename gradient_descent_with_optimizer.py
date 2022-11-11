import torch
import matplotlib.pyplot as plt

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

# Die Daten beschreiben offensichtlich die Gerade y = 2*x+1 (siehe oben)
# Wir tun nun so, als wuessten wir das nicht, und wollen via Gradientenabstieg 
# mithilfe von PyTorch die Parameter a,b der Geraden ax+b bestimmen.

# Parameter zufällig initialisieren:
# Im Kontext von Neuronalen Netzen oft auch als Gewichte (weights) bezeichnet
# Zwei zufällige Zahlen aus der Gleichverteilung [0,1]
weights = torch.rand(2, dtype=torch.float32, device=device) 

# Aufgabe 2a)
optimizer = 

for i in range(iterations):

    # Aufgabe 2a)
    # Gradienten im Optimizer zurücksetzen:

    # X hat die Dimension 10x2
    # weights hat Dimension 2
    # Für die Berchnung X*weights müssen wir daher noch eine Dimension zu 
    # weights hinzufuegen:
    w = weights.view(1, 2)  # analog zu numpy.reshape

    # Aufgabe 2b)
    # y=ax+b berechnen:
    y_estimate = 

    # Aufgabe 2c)
    # Fehler ("Loss") bezueglich der bekannten y-Werte berechnen:
    loss = 

    print("Iteration %d - Loss: %.4f" % (i, loss.item()))

    # Aufgabe 2d)
    # mit .backward() werden die Gradienten d loss / d weights berechnet
    
    # Aufgabe 2e)

# Daten und optimierte Gerade plotten:
plt.figure()
plt.plot(x.numpy(), y.numpy(), 'kx')
y_final = torch.sum(X*weights, dim=-1)
plt.plot(x.numpy(), y_final.detach().numpy(), 'g-')
plt.show()

    