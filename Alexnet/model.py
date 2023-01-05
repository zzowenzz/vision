import torch
import torch.nn.functional as F
import torch.nn as nn 
import torchinfo

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            nn.Conv2d(96, 256, kernel_size=5, padding=2),           
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            nn.Conv2d(256, 384, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

# test for shape
# net = AlexNet()
# report = torchinfo.summary(net, input_size=(1,1,224,224))
# summary_report = str(report)
# with open("architecture.txt", "w") as f:
#     f.write((summary_report))