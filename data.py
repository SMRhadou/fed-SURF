import numpy as np

from models.ResNet import *
from models.unrolledModels import *
from training import *

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

def loadDataset(train):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transforms.ToTensor())
    return trainset

def selectClasses(dataset, args, imbalance=False):
    nClasses = args.nClasses
    subDatasetSize = args.subDatasetSize
    subTrainSize = args.nTrainPerAgent*args.nAgents
    subValidSize = subDatasetSize - subTrainSize
    if imbalance:
        idx = np.random.permutation(nClasses)
        nValidClass = (subValidSize * np.array([0.4, 0.3, 0.3]))[idx].astype(int)      # number of samples per class
        nTrainClass = (subTrainSize * np.array([0.4, 0.3, 0.3]))[idx].astype(int)   # number of training samples per class
    else:
        nValidClass = int(subValidSize/nClasses) * np.ones((nClasses,), dtype=int)      # number of samples per class
        nTrainClass = int(subTrainSize/nClasses) * np.ones((nClasses,), dtype=int)   # number of training samples per class
    data_train = torch.empty((subTrainSize, dataset.data.shape[1], dataset.data.shape[2], dataset.data.shape[3]))
    data_valid = torch.empty((subValidSize, dataset.data.shape[1], dataset.data.shape[2], dataset.data.shape[3]))
    targets_train = torch.empty((subTrainSize,nClasses))
    targets_valid = torch.empty((subValidSize, nClasses))
    classes = np.random.choice(len(dataset.classes), nClasses, replace=False)  # select classes randomly without replacement
    for i in range(len(classes)):
        idx = np.where(dataset.targets == classes[i])[0]
        one_hot = F.one_hot(torch.arange(0, 3) % 3)
        data, targets = torch.tensor(dataset.data)[idx], one_hot[i].repeat(len(idx), 1)#i*torch.ones_like(torch.tensor(dataset.targets)[idx])
        nTrain = 0.8 * len(data)
        idx = np.random.randint(0, nTrain, nTrainClass[i])
        data_train[np.sum(nTrainClass[:i]):np.sum(nTrainClass[:i+1])], targets_train[np.sum(nTrainClass[:i]):np.sum(nTrainClass[:i+1])] = data[idx], targets[idx]
        idx = np.random.randint(nTrain, len(data), nValidClass[i])
        data_valid[np.sum(nValidClass[:i]):np.sum(nValidClass[:i+1])], targets_valid[np.sum(nValidClass[:i]):np.sum(nValidClass[:i+1])] = data[idx], targets[idx]
    if np.sum(nTrainClass) != subTrainSize:
        idx = np.random.randint(0, nTrain, subTrainSize-np.sum(nTrainClass))
        data_train[np.sum(nTrainClass):], targets_train[np.sum(nTrainClass):] = data[idx], targets[idx]
    if np.sum(nValidClass) != subValidSize:
        idx = np.random.randint(nTrain, len(data), subValidSize-np.sum(nValidClass))
        data_valid[np.sum(nValidClass):], targets_valid[np.sum(nValidClass):] = data[idx], targets[idx]
    data = np.concatenate((data_train, data_valid))
    targets = torch.cat((targets_train, targets_valid))
    shuffleidx = torch.randperm(subDatasetSize)       
    return data[shuffleidx], targets[shuffleidx]

def subDataset(dataset, args, outDist=False, imbalance=False):
    if outDist:
        transform = outDistTransform()
    else:
        transform = randTransform()
    data, targets = selectClasses(dataset, args, imbalance)
    dataTensor = torch.empty((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
    for i in range(dataTensor.shape[0]):
        dataTensor[i] = transform(data[i].astype(np.uint8))          # subDatasetSize x image size (3d)
    return dataTensor.float(), targets, transform

def createMetaDataset(model, dataset, args, outDist=False, imbalance=False):
    nTrainPerAgent = args.nTrainPerAgent
    metadataset = {}
    batchSize = 100
    for i in range(args.nDatasets):
        images, targets, transform = subDataset(dataset, args, outDist, imbalance)                    # nAgents*nExamples (=subDatasetSize) x image size (3d) (Ex: 1200x3x32x32)
        features = torch.empty((args.subDatasetSize, model.module.linear.in_features))                                                      # nAgents*nExamples x nFeatures (Ex: 1200x2048)
        for ibatch in range(images.shape[0]//batchSize):
            features[ibatch*batchSize:(ibatch+1)*batchSize] = model(images[ibatch*batchSize:(ibatch+1)*batchSize].to('cuda:0')) 
        distributedFeatures, distributedTargets = spreadDataAmongAgents(features, targets, args.nAgents)                                     # nAgents x nExamples x nFeatures (Ex: 100x12x2048)
        metadataset[i] = ((distributedFeatures[:,:nTrainPerAgent], distributedTargets[:,:nTrainPerAgent]), 
                            (distributedFeatures[:,nTrainPerAgent:], distributedTargets[:,nTrainPerAgent:]), transform)   # (train_dataset, test_dataset, transform)
    torch.cuda.empty_cache()
    return metadataset

def spreadDataAmongAgents(dataset, targets, nAgents):
    return torch.reshape(dataset, (nAgents, -1, dataset.shape[1])), torch.reshape(targets, (nAgents, -1, targets.shape[1])) # nAgents x nExamples(training+validation) x nFeatures 

def randTransform():
    idx = np.random.randint(0, 4)
    if idx == 0:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 1:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 2:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 3:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomRotation(90),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return transform


def outDistTransform():
    idx = np.random.randint(0, 4)
    if idx == 0:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5), transforms.RandomCrop(32, padding=4)], p=0.7),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 1:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5), transforms.RandomRotation(45)], p=0.7),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 2:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.GaussianBlur(kernel_size=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif idx == 3:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomRotation(180),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform

