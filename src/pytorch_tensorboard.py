from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision
import tensorboard as tb
import tensorflow as tf 
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

input_size = 784
num_classes = 10

epoch = 1

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
        
train_dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(),download=True)

step = 0
batch_sizes = [256]
lrs = [0.01]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for batch_idx, batch_size in enumerate(batch_sizes):
    for lr in lrs:
        
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        model.train()

        writer = SummaryWriter(f'runs/MNIST/MiniBatchSizeNew {batch_size} LRNew {lr}')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for _ in range(epoch):
            accuracies = []
            losses = []
            for idx, (data, target) in enumerate(train_loader):
            
                scores = model(data)
                loss = criterion(scores, target)
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scores = model(data)
                    
                _, pred = scores.max(1)
                
                num_correct = (pred==target).sum()
                
                running_acc = num_correct/data.shape[0]
                accuracies.append(running_acc)
                losses.append(loss.item())
                
                writer.add_scalar('Training Loss', loss, global_step=step)
                writer.add_scalar('Training Acc', running_acc, global_step=step)
                
                features = data.reshape(data.shape[0], -1)
                class_labels = [classes[label] for label in pred]
                image_grid = torchvision.utils.make_grid(data)
                writer.add_image('mnist_img', image_grid)
                writer.add_histogram('fc1', model.fc1.weight)
                
                if idx==230:
                    writer.add_embedding(features, metadata=class_labels, label_img=data, global_step=batch_idx)
                
                step+=1
            
            writer.add_hparams({'Lr': lr, 'bsize': batch_size},
                               {'aacuracy': sum(accuracies)/len(accuracies), 'loss': sum(losses)/len(losses)})
            

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