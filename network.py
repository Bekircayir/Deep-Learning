from torch import nn

# Aufgabe 1
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_conv1 = nn.Conv2d(3, 32, (3, 3), padding= 1 , stride = 1)
        self.cnn_act1 = nn.ReLU()
        self.cnn_conv2 = nn.Conv2d(32, 32, (3, 3), padding= 1 , stride = 1)
        # self.cnn_act2 = nn.ReLU()
        self.cnn_conv3 = nn.Conv2d(32, 32, (3, 3), padding= 1 , stride = 1)
        # self.cnn_act3 = nn.ReLU()
        self.cnn_conv4 = nn.Conv2d(32, 32, (3, 3), padding= 1 , stride = 1)
        # self.cnn_act4 = nn.ReLU()
        self.fcl = nn.Linear(32, 10)
        self.cnn_maxpool = nn.MaxPool2d((2, 2))
        # self.cnn_act_mpool = nn.ReLU()
        # self.cnn_avr_pool = nn.AvgPool2d(1)
        self.cnn_adpt = nn.AdaptiveAvgPool2d((1, 1))        
        # Netzstruktur definieren: 
        self.cnn_flt = nn.Flatten()


    def forward(self, x):
        # Diese Funktion definiert den Forward-Pass
        X = self.cnn_conv1(x)
        X = self.cnn_maxpool(X)
        X = self.cnn_act1(X)
        
        X = self.cnn_conv2(X)
        X = self.cnn_maxpool(X)
        X = self.cnn_act1(X)
        
        X = self.cnn_conv3(X)
        X = self.cnn_maxpool(X)
        X = self.cnn_act1(X)
        
        X = self.cnn_conv4(X)
        X = self.cnn_maxpool(X)
        X = self.cnn_act1(X)
        X = self.cnn_adpt(X)
        X = self.cnn_flt(X)
        X = self.fcl(X) 
        
        return X



# Aufgabe 5

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet6(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(1024, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
