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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class Net(nn.Module):
    def __init__(self, imgSize, FClayerNeurons, numbClasses, cLayerblockSize, numbConvLayers, numbFCLayers):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cLayers = nn.ModuleList()
        self.fcLayers = nn.ModuleList()
        self.cLayerblockSize = cLayerblockSize
        self.createLayers(numbConvLayers, numbFCLayers, imgSize, FClayerNeurons, numbClasses, cLayerblockSize)

    def createLayers(self, numbConvLayers, numbFCLayers, imgSize, FClayerNeurons, numbClasses, cLayerblockSize):
        finalOutChannels, cLayerNumBlocks = self.createConvLayers(numbConvLayers, cLayerblockSize)

        print(f"Convolutional layers: {len(self.cLayers)}")
        print(f"Output channels after last layer: {finalOutChannels}")
        print(f"Number of blocks: {cLayerNumBlocks}")
        self.createFCLayers(finalOutChannels, cLayerNumBlocks, imgSize, FClayerNeurons, numbClasses, numbFCLayers)
        return
    
    def createConvLayers(self, numbLayers, blockSize):
        self.cLayers.append(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1))
        inChannels  = 8
        outChannels = 8
        blocks = 0
        for i in range(numbLayers-1):
            if (i+1) % blockSize == 0 and i != 0:
                outChannels = outChannels * 2
                blocks += 1
            self.cLayers.append(nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1))
            inChannels = outChannels
        return outChannels, blocks
    
    def createFCLayers(self, finalOutChannels, cLayerNumBlocks, imgSize, layerNeurons, numbClasses, numbLayers):
        # cLayerNumBlocks is number of convolution blocks, which is also the number of pooling layers reducing the image size by 2 everytime
        finalImgSize = self.findFinalImgSize(imgSize)

        self.fcLayers.append(nn.Linear(finalImgSize, layerNeurons))

        # only do this if more than 2 layers are requested 2 are the minimum
        for i in range(numbLayers-2):
            self.fcLayers.append(nn.Linear(layerNeurons, layerNeurons))

        self.fcLayers.append(nn.Linear(layerNeurons, numbClasses))

    def findFinalImgSize(self, imgSize):
        dummyImg = torch.zeros(1, 3, imgSize, imgSize)
        x = dummyImg 

        for i, cLayer in enumerate(self.cLayers): 
            x = F.relu(cLayer(x)) 
            if (i + 1) % self.cLayerblockSize == 0: 
                x = self.pool(x) 

        in_features = x.view(1, -1).shape[1]
        return in_features

    def forward(self, x):

        for i, cLayer in enumerate(self.cLayers):
            x = F.relu(cLayer(x))

            if (i + 1) % self.cLayerblockSize == 0:
                x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # fclayers except last
        for fcLayer in self.fcLayers[:-1]:
            x = F.relu(fcLayer(x))

        # final layer
        x = self.fcLayers[-1](x)

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
        if numWorkers > 0:
            return DataLoader(
                trainData,
                batch_size=batchSize,
                num_workers=numWorkers,
                worker_init_fn=self.seed_worker,
                generator=g,
                persistent_workers=True 
            )
        else:
            return DataLoader(
                trainData,
                batch_size=batchSize,
                num_workers=numWorkers,
                worker_init_fn=self.seed_worker,
                generator=g,
            )

    def getLoss(self, logits, labels):

        lossFunc = nn.CrossEntropyLoss()
        loss = lossFunc(logits, labels)
        return loss
    
    def getError(self, logits, labels):

        outputs = torch.softmax(logits, dim=1)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct
        
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
    
    def test(self, testData, batchSize, numWorkers):
        testData = self.dataLoader(testData, batchSize, numWorkers)
        totalLoss = 0
        correct = 0
        samples = 0

        for inputs, labels in testData:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = self.nn.forward(inputs)

            loss = self.getLoss(logits, labels)
            totalLoss += loss.item() * inputs.size(0)

            correct += self.getError(logits, labels)

            samples += labels.size(0)

        avgLoss = totalLoss / samples
        accuracy = correct / samples
        print(f"Test Loss: {avgLoss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
        return avgLoss, accuracy



if __name__ == "__main__":

    imgSize = 32
    numbChannles = 3
    outputs = 10
    batchSize = 200
    epochs = 100
    trainNumWorkers = 3
    testNumWorkers = 0
    learningRate = 0.001


    net = Net(imgSize, FClayerNeurons=120, numbClasses=10, cLayerblockSize=8, numbConvLayers=2, numbFCLayers=2)
    trainer = Tester(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), learningRate, momentum=0.9)


   

    trainer.train(train_data, epochs, batchSize, trainNumWorkers)
    trainer.test(test_data, batchSize, testNumWorkers)
