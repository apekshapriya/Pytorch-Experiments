import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F

input_size = 784
num_classes = 10
batch_size = 64
lr = 0.001
epoch = 5


class CNN(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=10):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))  # (W-2P-K)/s  +  1  Same conv (28,28)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))                                                                           # (14,14)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels= 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same conv            # (14,14)
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1(x)
        return x
        
    
model = CNN()
# x = torch.rand((64,1,28,28))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= lr)


train_dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


for _ in range(epoch):
    for idx, (data, target) in enumerate(train_loader):
        
       
        # data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        loss = criterion(scores, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        

def check_aacuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            
            # data = data.reshape(data.shape[0], -1)
            
            scores = model(data)
            
            _, pred = scores.max(1)
            
            num_correct += (pred==target).sum()
            num_samples += pred.size()[0]
    
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100:.2f}')
    

check_aacuracy(train_loader, model)
check_aacuracy(test_loader, model)