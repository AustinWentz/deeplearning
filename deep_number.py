# Use neural network training on MNIST handdrawn number set
# Predict with accuracy the handdrawn number drawn

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

import math as m

# Set image size
dimension = 28
area = int(m.pow(dimension, 2))

# Set output size
output = 64

# Class size
categories = 10

# Download train and test data
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

# Load train and test data
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# Neural Network
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(area, output)
        self.fc2 = nn.Linear(output, output)
        self.fc3 = nn.Linear(output, output)
        self.fc4 = nn.Linear(output, categories)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

# Instantiate neural network
net = Net()

# Init optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the amount of times to run through the data set
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featureset and labels
        X, y = data

        # Zero the gradients after each batch
        net.zero_grad()

        # Run the batch through the network
        result = net(X.view(-1, area))

        # Calculate loss, use nll because data is a scalar value
        loss = F.nll_loss(result, y)
        loss.backward()
        optimizer.step()

    print(loss)

# Test Model Accuracy
correct = 0
total = 0

with torch.no_grad():
    for traindata in trainset:
        X, y = traindata
        result = net(X.view(-1, area))
        for idx, i in enumerate(result):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
    print("Trainset Model Accuracy: ", round(correct/total, 3))

    correct = 0
    total = 0

    for testdata in testset:
        X, y = testdata
        result = net(X.view(-1, area))
        for idx, i in enumerate(result):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
    print("Testset Model Accuracy: ", round(correct/total, 3))