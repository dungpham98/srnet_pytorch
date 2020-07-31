"""
Implementation of the Paper "Deep Residual Network for Steganalysis of Digital Images"
http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# decay rate = 1 - momemtum = 0.9 by default

class LayerT1(nn.Module):
    """
    Building block Tier 1 used in Srnet. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(LayerT1, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
    

class LayerT2(nn.Module):
    """
    Building block Tier 2 used in Srnet
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(LayerT2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out)
        )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        x += residual 
        return x


class LayerT3(nn.Module):
    """
    Building block Tier 3 used in Srnet.
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(LayerT2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out)
        )

        self.layer_res = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride = 2, padding=1),
            nn.BatchNorm2d(channels_out)
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        residual = self.layer_res(x)
        x = self.layers(x)
        x += residual
        return x 

class LayerT4(nn.Module):
    """
    Building block Tier 4 used in Srnet
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(LayerT2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out)
        )

    def forward(self, x):
        avgp = torch.mean(x, dim=(2,3), keepdim=True)
        return avgp

class Srnet(nn.Module):
    def __init__(self):
        super(Srnet,self).__init__()
        layers = [ConvBNRelu(1,64)]
        layers.append(ConvBNRelu(64,16))

        for _ in range(5):
            layer = LayerT2(16,16)
            layers.append(layer)

        layers.append(LayerT3(16,16))
        layers.append(LayerT3(16,64))
        layers.append(LayerT3(64,128))
        layers.append(LayerT3(128,256))

        layers.append(LayerT4(256,512))
        # Fully Connected layer
        self.fc = nn.Linear(512*1*1, 2)
        
    def foward(self,x):
        flatten = x.view(x.size(0),-1)
        fc = self.fc(flatten)
        # print("FC:",fc.shape)
        out = F.log_softmax(fc, dim=1)
        return fc