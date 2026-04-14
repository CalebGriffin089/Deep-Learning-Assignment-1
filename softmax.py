import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np, random
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



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(imgSize*numbChannles, outputs),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def trainDataLoader(self, trainData, batchSize, numWorkers):
        return DataLoader(
            trainData,
            batch_size=batchSize,
            num_workers=numWorkers,
            worker_init_fn=self.seed_worker,
            generator=g,
        )
    
    def seed_worker(self, worker_id):
       
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return worker_seed
    
    def train(self, trainData, epochs, batchSize, numWorkers):
        for i in range(epochs):
            loss = self.trainOneEpoch(trainData, batchSize, numWorkers)
            print(f"Epoch: {i+1}, Loss: {sum(loss)/len(loss)}")
        return

    def trainOneEpoch(self, trainData, batchSize, numWorkers):
        runningLoss = 0.
        lastLoss = 0.
        allLoss = []

        for i, data in enumerate(self.trainDataLoader(trainData, batchSize, numWorkers)):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = self.forward(inputs)
            loss = self.getLoss(logits, labels)
            runningLoss += loss.item()

            loss.backward()
            optimizer.step()

            
            if i % 100 == 99:
                lastLoss = runningLoss / 100 
                allLoss.append(lastLoss)
                print('  batch {} loss: {}'.format(i + 1, lastLoss))
                runningLoss = 0.

        return allLoss
    
    def getLoss(self, logits, labels):

        lossFunc = nn.CrossEntropyLoss()
        loss = lossFunc(logits, labels)
        return loss
    
    def getError(self, logits, labels):

        outputs = torch.softmax(logits, dim=1)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct
    
    def test(self, testData, batchSize):
        testData = DataLoader(testData, batch_size=batchSize, shuffle=False)
        totalLoss = 0
        correct = 0
        samples = 0

        for inputs, labels in testData:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = self.forward(inputs)

            loss = self.getLoss(logits, labels)
            totalLoss += loss.item() * inputs.size(0)

            correct += self.getError(logits, labels)

            samples += labels.size(0)

        avgLoss = totalLoss / samples
        accuracy = correct / samples
        print(f"Test Loss: {avgLoss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
        return avgLoss, accuracy

imgSize = 32*32
numbChannles = 3
outputs = 10
batchSize = 50
epochs = 10
numWorkers = 1
learningRate = 0.001


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learningRate)
    model.train(train_data, epochs, batchSize, numWorkers)

    testLoss, testAcc = model.test(test_data, batchSize)
