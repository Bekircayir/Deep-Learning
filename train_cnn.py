from tabnanny import check
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from network import *

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    

    # Beispielbilder anzeigen
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        plt.show()

    # Wähle zufällig Trainingsbilder
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # Erstelle Netz
    net = Net()

    # Erstelle Verlustfunktion und Optimierer
    ...

    # Training
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            ...

            # forward + backward + optimize
            ...

            # print statistics
            ...
            if i % 2000 == 1999:    # print every 2000 mini-batches
                ...    

    print('Finished Training')

    # Speicher Neuronales Netz
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)