import torch
import torch.nn.functional as F
import torch.nn as nn 
import torchinfo

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = self.pool1(x)            
        x = F.relu(self.conv2(x))    
        x = self.pool2(x)            
        x = x.view(-1, 32*5*5)       
        x = F.relu(self.fc1(x))      
        x = F.relu(self.fc2(x))     
        x = self.fc3(x)              
        return x

# test for shape
lenet = Lenet()
report = torchinfo.summary(lenet, input_size=(1,1,32,32))
summary_report = str(report)
with open("architecture.txt", "w") as f:
    f.write((summary_report))