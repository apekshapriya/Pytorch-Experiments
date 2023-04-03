import torch

import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms

#Methods for dealing with class imbalance


# Class Weighting in loss, giving more priority to class which is minor

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))

#1. Oversampling by class weighting

# Create Random Data

batch_size = 5
shape = 20,10,1
arr = np.random.random((20,5,1))

#Create imbalanced labels
label = np.ones((20), dtype=int)

label[0] = 0
label[5] = 0
label[2] = 0

# Create custom dataset

class CustomDataset(Dataset):
    
    def __init__(self, arr, label, transform=None):
        
        self.arr = arr
        self.label = label
        self.transform = transform
        
    
    def __len__(self):
        return len(self.arr)
    
    
    def __getitem__(self, index):
        
        arr = self.arr[index]
        label = self.label[index]
        
        if self.transform:
            arr = self.transform(arr)
            
        return [arr, label]
        
    
my_transform = transforms.ToTensor()   
    
dataset = CustomDataset(arr, label, transform=my_transform)
    
classes = np.unique(label, axis=-1)

class_weights = {}

for  cl in classes:
    class_weights[cl] = 1/(len(np.where(cl==label)[0]))

sample_weights = [0] * len(dataset)
for ind, (data, label) in enumerate(dataset):
        
    class_weight = class_weights[label]
    sample_weights[ind] = class_weight

 
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

for ind, (data, label) in enumerate(loader):
    
    print(label)
