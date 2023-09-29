import numpy as np

from models.ResNet import *
from models.unrolledModels import *
from core.training import *

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

def loadDataset(train, Dataset='CIFAR10'):
    if Dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transforms.ToTensor())
    elif Dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root='./data', train=train, download=True, transform=transforms.ToTensor())
    elif Dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    return trainset

def classPermutation(classes, nClasses, nPerm):
    P = np.zeros((nPerm, nClasses), dtype=int)
    for i in range(nPerm):
        P[i] = np.random.choice(len(classes), nClasses, replace=False)
    return P

'''
1. cretate a distribution of classes (a list of occurances) and store it
2. construct this dataset based on the distribution
3. divide it uniformly among agents
'''
def classDistribution(classes):
    p = np.random.randint(1, 100, len(classes))
    return p/np.sum(p)

def subDataset(dataset, args, outDist=False):
    if outDist:
        transform = outDistTransform(args)
    else:
        transform = randTransform(args)
    classDist = classDistribution(dataset.classes) 
    data, targets = buildsubDataset(dataset, args, classDist)
    if args.Dataset == 'CIFAR10':
        dataTensor = torch.empty((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
    elif args.Dataset == 'MNIST':
        dataTensor = torch.empty((data.shape[0], 1, data.shape[1], data.shape[2]))
    for i in range(dataTensor.shape[0]):
        dataTensor[i] = transform(data[i].astype(np.uint8))          # subDatasetSize x image size (3d)
    return dataTensor.float(), targets, transform

def createMetaDataset(model, dataset, args, outDist=False, test=False):
    nDatasets = args.nDatasets if not test else 30
    nTrainPerAgent = args.nTrainPerAgent
    metadataset = {}
    batchSize = 100
    for i in range(nDatasets):
        images, targets, transform = subDataset(dataset, args, outDist)                    # nAgents*nExamples (=subDatasetSize) x image size (3d) (Ex: 1200x3x32x32)
        if args.mode == '0':
            features = torch.empty((args.subDatasetSize, model.module.linear.in_features))                                                      # nAgents*nExamples x nFeatures (Ex: 1200x2048)
            for ibatch in range(images.shape[0]//batchSize):
                features[ibatch*batchSize:(ibatch+1)*batchSize] = model(images[ibatch*batchSize:(ibatch+1)*batchSize].to('cuda')) 
        else:
            features = images
        distributedFeatures, distributedTargets = spreadDataAmongAgents(features, targets, args.nAgents, args)                                     # nAgents x nExamples x nFeatures (Ex: 100x12x2048)
        metadataset[i] = ((distributedFeatures[:,:nTrainPerAgent], distributedTargets[:,:nTrainPerAgent]), 
                            (distributedFeatures[:,nTrainPerAgent:], distributedTargets[:,nTrainPerAgent:]), transform)   # (train_dataset, test_dataset, transform)
    torch.cuda.empty_cache()
    return metadataset

def buildsubDataset(dataset, args, classDist):
    nClasses = len(classDist)#args.nClasses
    subDatasetSize = args.subDatasetSize
    subTrainSize = args.nTrainPerAgent*args.nAgents
    subValidSize = subDatasetSize - subTrainSize
    nValidClass = (subValidSize * classDist).astype(int)     # number of samples per class
    nTrainClass = (subTrainSize * classDist).astype(int)   # number of training samples per class
    if args.Dataset == 'CIFAR10':
        data_train = torch.empty((subTrainSize, dataset.data.shape[1], dataset.data.shape[2], dataset.data.shape[3]))
        data_valid = torch.empty((subValidSize, dataset.data.shape[1], dataset.data.shape[2], dataset.data.shape[3]))
    elif args.Dataset == 'MNIST':
        data_train = torch.empty((subTrainSize, dataset.data.shape[1], dataset.data.shape[2]))
        data_valid = torch.empty((subValidSize, dataset.data.shape[1], dataset.data.shape[2]))
    targets_train = torch.empty((subTrainSize, nClasses))
    targets_valid = torch.empty((subValidSize, nClasses))
    # classes = np.random.choice(len(dataset.classes), nClasses, replace=False)  # select classes randomly without replacement
    for i in range(len(classDist)):
        if args.Dataset == 'CIFAR10':
            idx = np.where(dataset.targets == np.array([i]))[0]
        elif args.Dataset == 'MNIST':
            idx = np.where(dataset.targets == i)[0]
        one_hot = F.one_hot(torch.arange(0, nClasses) % nClasses)
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

# def subDataset(dataset, args, classes, outDist=False):
#     if outDist:
#         transform = outDistTransform()
#     else:
#         transform = randTransform()
#     data, targets = buildsubDataset(dataset, args, classes)
#     dataTensor = torch.empty((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
#     for i in range(dataTensor.shape[0]):
#         dataTensor[i] = transform(data[i].astype(np.uint8))          # subDatasetSize x image size (3d)
#     return dataTensor.float(), targets, transform

# def createMetaDataset(model, dataset, args, classesDist=None, outDist=False, test=False):
#     nDatasets = args.nDatasets if not test else 30
#     nTrainPerAgent = args.nTrainPerAgent
#     metadataset = {}
#     batchSize = 100
#     if classesDist is None:
#         classesDist = classPermutation(dataset.classes, args.nClasses, 100)
#     for i in range(nDatasets):
#         idx = np.random.randint(0, classesDist.shape[0], 1)[0]
#         images, targets, transform = subDataset(dataset, args, classesDist[idx], outDist)                    # nAgents*nExamples (=subDatasetSize) x image size (3d) (Ex: 1200x3x32x32)
#         features = torch.empty((args.subDatasetSize, model.module.linear.in_features))                                                      # nAgents*nExamples x nFeatures (Ex: 1200x2048)
#         for ibatch in range(images.shape[0]//batchSize):
#             features[ibatch*batchSize:(ibatch+1)*batchSize] = model(images[ibatch*batchSize:(ibatch+1)*batchSize].to('cuda:0')) 
#         distributedFeatures, distributedTargets = spreadDataAmongAgents(features, targets, args.nAgents)                                     # nAgents x nExamples x nFeatures (Ex: 100x12x2048)
#         metadataset[i] = ((distributedFeatures[:,:nTrainPerAgent], distributedTargets[:,:nTrainPerAgent]), 
#                             (distributedFeatures[:,nTrainPerAgent:], distributedTargets[:,nTrainPerAgent:]), transform)   # (train_dataset, test_dataset, transform)
#     torch.cuda.empty_cache()
#     return metadataset, classesDist

def spreadDataAmongAgents(dataset, targets, nAgents, args):
    if args.mode == '0':
        return torch.reshape(dataset, (nAgents, -1, dataset.shape[1])), torch.reshape(targets, (nAgents, -1, targets.shape[1])) # nAgents x nExamples(training+validation) x nFeatures 
    else:
        return torch.reshape(dataset, (nAgents, -1, dataset.shape[1]*dataset.shape[2]*dataset.shape[3])), torch.reshape(targets, (nAgents, -1, targets.shape[1]))

def randTransform(args):
    idx = np.random.randint(0, 4)
    if args.Dataset == 'CIFAR10':
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
    elif args.Dataset == 'MNIST':
        if idx == 0:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomCrop(28, padding=4),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.1307], [0.3081])])
        elif idx == 1:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.1307], [0.3081])])
        elif idx == 2:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.1307], [0.3081])])
        elif idx == 3:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomRotation(90),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.1307], [0.3081])])

    return transform


def outDistTransform(args):
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

