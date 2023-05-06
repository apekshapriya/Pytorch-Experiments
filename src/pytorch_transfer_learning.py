import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision

epochs = 2
learning_rate = 0.001
batch_size = 512

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
model = torchvision.models.vgg16(pretrained=False)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

criterion = nn.CrossEntropyLoss()

model.avgpool = Identity()  # giving 1*1*512 from (7*7*512)
# model.classifier = Identity()

model.classifier = nn.Sequential(nn.Linear(512*7*7, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 10))
# print(model)

x = torch.rand((64, 3,32,32))
print(model(x).shape)


train_dataset = datasets.CIFAR10(root='dataset',train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='dataset',train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)


for epoch in range(epochs):
    
    for idx, (data, target) in enumerate(train_loader):
        
        scores = model(data)
        
        loss = criterion(scores, target)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    

def check_accuracy(loader, model):
    
    model.eval()
    
    with torch.no_grad():
        
        num_correct = 0
        num_samples = 0
        for idx, (data, target) in enumerate(loader):
            
            
            scores = model(data)
            
            _, pred = scores.max(1)
            
                   
            num_correct += sum(pred==target)
            num_samples +=len(pred)
            
        
    print(f'Accuracy is: {num_correct/num_samples}')
    
    
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)