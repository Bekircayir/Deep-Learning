# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Daten und Visualisierung
import matplotlib.pyplot as plt
import numpy as np

# Aufgabe A
x = np.random.rand(100)
y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(100)*0.8

# Plotten Sie die Beispieldaten
plt.scatter(x,y)
# plt.show()

#_____________

# Aufgabe B

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       
       # Implementieren Sie einen linearen Layer mit 1 Input und 1 Output
       self.linlayer = nn.Linear(1, 1)

   def forward(self, x):

       # Implementieren Sie die Berechnung des Outputs mit dem Input 
       x = self.linlayer(x)  
       return x

net = Net()
print(net)

#_____________

# convert numpy array to tensor in shape of input size
x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()


# print(net(x))

#_____________

# Aufgabe C

# Definieren Sie den Optimizer und die Loss Function
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
loss_func = nn.MSELoss()

#_____________

# Aufgabe D

# Implementieren Sie den Lernalgorithmus (Gradient Descent) mit 250 Epochen
epochs = 250

for i in range(250):

   # Berechne Vorhersage
   prediction = net(x)

   # Berechne Loss Funktion
   loss = loss_func(prediction,y)

   # Update
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()   

   if i % 10 == 0:
       # plot and show learning process
       plt.cla()
       plt.scatter(x.data.numpy(), y.data.numpy())
       plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
       plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
       plt.pause(0.1)

plt.show()

#_____________
