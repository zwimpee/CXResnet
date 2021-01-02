"""
This file contains the components and framework for 
the model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Class for defining stack of layers.
class LayerStack(nn.Module):
    """
    Create an object to apply at least one layer to input, with 
    the option to apply an additional set of layers to the output
    of the 'top' layer. 
    
    Note: The arg values for any additional layers must be explicitly 
    defined before using them in creating an instance of this object. 
    The keyword arguments will only be passed to the 'top' layer.
    """
    def __init__(self, top ,layers = None,*args, **kwargs):
        super(LayerStack, self).__init__()
        
        # Defining the required top layer.
        self.top = top(*args, **kwargs)
        
        # Set parameter has_outlayer to false by default.
        self.layers = layers
        
        # Create Sequential layer if num. layers > 1.
        if type(layers) is list:
            self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.top(x)
        if self.layers is not None:
            x = self.layers(x)
        return x

    
    

# Class for implementing a residual connection.
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualLayer, self).__init__()
        
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        
        self.block = LayerStack(
            nn.Conv2d,layers=[
                nn.BatchNorm2d(self.out_channels),nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.out_channels,
                          out_channels=self.out_channels,
                          kernel_size=3,stride=1,
                          padding=1,bias=False),
                nn.BatchNorm2d(self.out_channels)],
            in_channels=self.in_channels,out_channels=self.out_channels,
            kernel_size=3,stride=self.stride,padding=1,bias=False)
        
        if self.in_channels != self.out_channels:
            self.shortcut = LayerStack(
                nn.Conv2d,layers=nn.BatchNorm2d(self.out_channels),
                in_channels=self.in_channels,out_channels=self.out_channels,
                kernel_size=1,stride=self.stride,bias=False)
        else:
            self.shortcut = None
            
    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out += identity
        return out
    


    
    
# Simple convolutional network.
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        l1 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(16)]
        self.conv1 = LayerStack(nn.Conv2d,layers=l1,in_channels=3,out_channels=16,kernel_size=3,padding = 1)
        
        l2 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(64)]
        self.conv2 = LayerStack(nn.Conv2d,layers=l2,in_channels=16,out_channels=64,kernel_size=3,padding = 1)
        
        l3 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(128)]
        self.conv3 = LayerStack(nn.Conv2d,layers=l3,in_channels=64,out_channels=128,kernel_size=3,padding = 1)
        
        l4 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(256)]
        self.conv4 = LayerStack(nn.Conv2d,layers=l4,in_channels=128,out_channels=256,kernel_size=3,padding = 1)
        
        l5 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(512)]
        self.conv5 = LayerStack(nn.Conv2d,layers=l5,in_channels=256,out_channels=512,kernel_size=3,padding = 1)
        
        
        self.fc1 = LayerStack(nn.Linear, layers = [nn.ReLU(),nn.BatchNorm1d(64)], in_features = 512*7*7, out_features = 64)
        self.classifier = LayerStack(nn.Linear, in_features=64, out_features=3)
     
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fc1(x)
        x = self.classifier(x)
        return x
    
    
    
# Simple network containing residual connection layers.
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        
        l1 = [nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.BatchNorm2d(16)]
        self.conv1 = LayerStack(nn.Conv2d,layers=l1,in_channels=3,out_channels=16,kernel_size=3,padding = 1)
        
        self.layer1 = ResidualLayer(in_channels=16, out_channels=64, stride=2)
        self.layer2 = ResidualLayer(in_channels=64, out_channels=128, stride=2)
        self.layer3 = ResidualLayer(in_channels=128, out_channels=256, stride=2)
        self.layer4 = ResidualLayer(in_channels=256, out_channels=512, stride=2)
        
        
        
        self.fc1 = LayerStack(nn.Linear, layers = [nn.ReLU(),nn.BatchNorm1d(64)], in_features = 512*7*7, out_features = 64)
        self.classifier = LayerStack(nn.Linear, in_features=64, out_features=3)
        
        
        
        # Initialize the layer weights 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
     
     
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fc1(x)
        x = self.classifier(x)
        return x