import torch
import torchvision
import torchvision.transforms as transforms
from network import *

if __name__ == '__main__':

    # Definiere Transformation der Inputs
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 Dataset
    batch_size = 10000
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # Labels
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    
    
    # Lade Neuronales Netz
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    # Berechne Genauigkeit
    for batch_idx, (X, y) in enumerate(testloader):
        pred = net(X)
        pred = torch.argmax(pred, dim=-1)
        accuracy = torch.mean((pred == y).float()).item()

    print(f'Accuracy of the network on the 10000 test images: {100 * accuracy} %')