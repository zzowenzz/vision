import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn as nn
from model import vgg11, vgg13, vgg16, vgg19
import pandas as pd
import time
import multiprocessing
import os 

# Train network on single gpu or cpu
def try_gpu(i=0): 
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# Load data and do data transformation
def load_data(batch_size):
    trans = {
        "train": transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    }
    train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans["train"], download=True)
    test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans["test"], download=True)
    return (data.DataLoader(train, batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()),
            data.DataLoader(test, batch_size, shuffle=False,num_workers=multiprocessing.cpu_count()), 
            len(train), 
            len(test))
# Hyper-parameter: batch_size, lr, num_epochs
batch_size = 128
lr, num_epochs = 0.1, 10
best_acc, save_path = 0.0, "best.pth"

# Prepare all parts: network, device, data iterator, network initialization, optimizer, loss function
net = vgg13()
net_name = "VGG13"
device = try_gpu()
train_iter, test_iter, num_train, num_test = load_data(batch_size)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# Create empty dataframe to record the training record
if not os.path.exists(net_name+".csv"):
    df = pd.DataFrame(columns=["Network", "Parameter", "Dataset", "Epoch", "Device", "Time cost(sec)", "Batch size", "Lr","Best test acc"])
df = pd.read_csv(os.getcwd()+"/"+net_name+".csv")

# Train
print("Train {} on {}".format(net_name, device))
print("{} images for training, {} images for validation\n".format(num_train,num_test))
total_time = 0.0
for epoch in range(num_epochs):
    batch_time = time.time()
    train_loss, train_acc, test_acc = 0.0, 0.0, 0.0
    net.train()
    # For each batch 
    for i, (x, y) in enumerate(train_iter):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
    
        with torch.no_grad():
            train_loss += l/batch_size
            train_acc += (y_hat.argmax(axis=1) == y).sum().item()
    train_acc /= num_train

    net.eval()
    with torch.no_grad():
        # For each batch
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            test_acc += (net(X).argmax(axis=1) == y).sum().item()
    test_acc /= num_test
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(net.state_dict(), save_path)
    time_end = time.time()
    total_time += (time_end - batch_time)
    print("Epoch {}, train_loss {}, train_acc {}, best_acc {}, test_acc {}, time cost {} sec".format(epoch+1, "%.4f" % train_loss, "%.2f" % train_acc, "%.2f" %best_acc, "%.2f" %test_acc,  "%.2f" %(time_end - batch_time)))
# number of parameter
with open("vgg13.txt", "r") as f:
    for line in f:
        if "Total params: " in line:
            num_para = line.split()[-1]
df = pd.concat([df, pd.DataFrame.from_records([{"Network":net_name, "Parameter": num_para, "Dataset":"FashionMNIST", "Epoch":num_epochs, "Device":torch.cuda.get_device_name(0), "Time cost(sec)": "%.1f" %total_time , "Batch size":batch_size, "Lr":lr, "Best test acc":best_acc}])])
df.to_csv(net_name+".csv",index=False,header=True)
print("\nFinish training")

