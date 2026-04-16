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
        self.createConvLayers(numbConvLayers, cLayerblockSize)
        self.createFCLayers(imgSize, FClayerNeurons, numbClasses, numbFCLayers)
        return
    
    def createConvLayers(self, numbLayers, blockSize):
        inChannels  = 8
        outChannels = 8
        blocks = 0

        self.cLayers.append(nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1), nn.BatchNorm2d(outChannels), nn.ReLU()))

        for i in range(numbLayers-1):

            if (i+1) % blockSize == 0 and i != 0:
                outChannels = outChannels * 2
                blocks += 1

            self.cLayers.append(nn.Sequential(nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1), nn.BatchNorm2d(outChannels), nn.ReLU()))
            nn.BatchNorm2d(outChannels)
            inChannels = outChannels
    
    def createFCLayers(self, imgSize, layerNeurons, numbClasses, numbLayers):
        # cLayerNumBlocks is number of convolution blocks, which is also the number of pooling layers reducing the image size by 2 everytime
        finalImgSize = self.findFinalImgSize(imgSize)

        self.fcLayers.append(nn.Sequential(nn.Linear(finalImgSize, layerNeurons), nn.ReLU()))

        # only do this if more than 2 layers are requested 2 are the minimum
        for i in range(numbLayers-2):
            self.fcLayers.append(nn.Sequential(nn.Linear(layerNeurons, layerNeurons), nn.ReLU()))

        self.fcLayers.append(nn.Sequential(nn.Linear(layerNeurons, numbClasses), nn.ReLU()))

    def findFinalImgSize(self, imgSize):
        self.eval()
        with torch.no_grad():
            dummyImg = torch.zeros(1, 3, imgSize, imgSize)
            x = dummyImg 

            for i, cLayer in enumerate(self.cLayers): 
                x = cLayer(x)

                if (i + 1) % self.cLayerblockSize == 0: 
                    x = self.pool(x) 

            in_features = x.view(1, -1).shape[1]
            self.train()
            return in_features
        

    def forward(self, x):

        for i, cLayer in enumerate(self.cLayers):
            x = cLayer(x)

            if (i + 1) % self.cLayerblockSize == 0:
                x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        for fcLayer in self.fcLayers[:-1]:
            x = fcLayer(x)

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

    def saveData(self, data, train):
        with open(fileName, "a") as f:
            if not train:
                f.write(f"Training Done: \n Loss, Accuracy (%) \n")
            for i in data:
                if i == data[-1]:
                    f.write(f"{i}\n")
                else:
                    f.write(f"{i},")

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

        for epoch in range(epochs):  

            running_loss = 0.0
            for i, data in enumerate(dataLoader, 0):
                
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                
                optimizer.zero_grad()

                
                outputs = self.nn.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    data = [epoch, i+1, running_loss/100]
                    self.saveData(data, True)
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
        data  = [avgLoss, accuracy]
        self.saveData(data, False)
        print(f"Test Loss: {avgLoss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
        return avgLoss, accuracy



if __name__ == "__main__":
    fileName  = "output.txt"
    imgSize = 32
    numbChannles = 3
    outputs = 10
    batchSize = 200
    epochs = 20
    trainNumWorkers = 4
    testNumWorkers = 0
    learningRate = 0.1
    with open(fileName, "a") as f:
        f.write(f"Options:\n Epochs: {epochs}, BatchSize: {batchSize}, Loss:,  Learning Rate: {learningRate} \n")

    net = Net(imgSize, FClayerNeurons=120, numbClasses=10, cLayerblockSize=4, numbConvLayers=16, numbFCLayers=4)
    trainer = Tester(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), learningRate, momentum=0.9)


   

    trainer.train(train_data, epochs, batchSize, trainNumWorkers)
    trainer.test(test_data, batchSize, testNumWorkers)
