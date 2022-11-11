import torch
import matplotlib.pyplot as plt

iterations = 10
learning_rate = 0.01
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
weights = torch.rand(2, dtype=torch.float32, device=device) 
# Für den Gradientenabstieg muessen wir PyTorch sagen, 
# dass wir die Gradienten von weights berechnen wollen:
weights = weights.requires_grad_()

# Daten und initiale Gerade plotten:
plt.figure()
plt.plot(x.numpy(), y.numpy(), 'kx')
y_init = torch.sum(X*weights, dim=-1)
plt.plot(x.numpy(), y_init.detach().numpy(), 'r-')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

for i in range(iterations):
    # X hat die Dimension 10x2
    # weights hat Dimension 2
    # Für die Berchnung X*weights müssen wir daher noch eine Dimension zu 
    # weights hinzufuegen:
    w = weights.view(1, 2)  # so wie numpy.reshape

    # Aufgabe 1a)
    # y=ax+b berechnen:
    y_estimate = 

    # Aufgabe 1b)
    # Fehler ("Loss") bezueglich der bekannten y-Werte berechnen:
    loss = 

    print("Iteration %d - Loss: %.4f" % (i, loss.item()))

    # Aufgabe 1c)
    # mit .backward() werden die Gradienten d loss / d weights berechnet
    

    print("Gradienten: ", weights.grad)

    # Aufgabe 1d)
    # Ein einzelner Schritt des Gradientenabstiegs:
    weights = 

    # Nach der obigen Operation ist weights kein "Leaf Tensor" mehr.
    # Um im nächsten Schritt wieder die Gradienten zu berechnen, machen wir
    # aus weights wieder einen Leaf Tensor:
    weights = weights.detach().requires_grad_()
    # .detach(): erstellt einen neuen Tensor, der von der Gradientenberechnung entkoppelt ist 
    # .requires_grad_(): erstellt einen neuen Tensor, für den Gradienten berechnet werden sollen

    print("Neue Parameter: ", weights)

# Daten und optimierte Gerade plotten:
plt.figure()
plt.plot(x.numpy(), y.numpy(), 'kx')
y_final = torch.sum(X*weights, dim=-1)
plt.plot(x.numpy(), y_final.detach().numpy(), 'g-')
plt.show()

    