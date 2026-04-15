import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np, random
import torch.optim as optim
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

seed = 0
g = torch.Generator()
g.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Tester():
    def __init__(self, network):
        self.nn = network.to(device)

    def seed_worker(self, worker_id):
       
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return worker_seed

    def dataLoader(self, trainData, batchSize, numWorkers):
        return DataLoader(
            trainData,
            batch_size=batchSize,
            num_workers=numWorkers,
            worker_init_fn=self.seed_worker,
            generator=g,
            persistent_workers=True 
        )
    
    def train(self, data, epochs, batchSize, trainNumWorkers):

        dataLoader = self.dataLoader(data, batchSize, trainNumWorkers)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataLoader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.nn.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        


if __name__ == "__main__":
    net = Net()
    trainer = Tester(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    imgSize = 32*32
    numbChannles = 3
    outputs = 10
    batchSize = 200
    epochs = 10
    trainNumWorkers = 4
    testNumWorkers = 0
    learningRate = 0.001

    trainer.train(train_data, epochs, batchSize, trainNumWorkers)
