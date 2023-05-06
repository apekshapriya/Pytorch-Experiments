import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

vgg16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512,512,512, "M"]

class VGG16(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=1000):
        """
            Initialized the conv and linear layers.
        Args:
            in_channels (int, optional):  Defaults to 3.
            num_classes (int, optional):  Defaults to 1000.
        """
        
        super(VGG16, self).__init__()
        
        self.conv_layers = self.create_conv_layers(in_channels, vgg16)
        self.Linear = nn.Sequential(nn.Linear(512*7*7, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, num_classes))
        
    def forward(self, x:torch.Tensor):
        """ Forward Pass

        Args:
            x (torch.Tensor): input images

        Returns:
            torch.Tensor: target for each input image
        """
        
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.Linear(x)
        
        return x
        
    def create_conv_layers(self, in_channels=3, architecture=[]):
        """Create the conv layer using the architecture given

        Args:
            in_channels (int, optional): _description_. Defaults to 3.
            architecture (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        layers = []
        for x in architecture:
            print(in_channels, x)
            
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=(1,1), stride=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        return nn.Sequential(*layers)
                

# Test the output shapes of the model

if __name__ == """__main__""":
                
    model = VGG16() 
    # print(model.conv_layers)
    x = torch.rand((1,3,224,224))

    print(model(x).shape)

        
        