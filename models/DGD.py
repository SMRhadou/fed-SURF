import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from core.allObjectives import compute_grad

def accuracy(last_layer, images, targets, device):
    acc = 0
    if len(last_layer.shape) < len(images.shape):
        last_layer = torch.reshape(last_layer, (last_layer.shape[0], images.shape[-1]+1, -1))
    logits =  (images.float().to(device) @ last_layer[:,:-1] + last_layer[:, -1, None]).cpu()
    outputs = torch.max(logits, axis=2)[1]
    for i in range(outputs.shape[0]):
        acc += torch.sum(outputs[i] == torch.argmax(targets[i], dim=1))
    return acc/outputs.numel()


def DGD(dataset, Graph, inner_loss, criterion, **kwargs):
    if "device" in kwargs.keys():
        device = kwargs["device"]
    else:
        device = "cuda"

    # Random Initialization
    y = torch.distributions.Normal(0.0, 5).sample((kwargs['nAgents'], kwargs['LLSize'])).float().to(device)
    
    # Forward path
    acc = []
    acc.append(accuracy(y, dataset[1][0], dataset[1][1].long(), device).item())
    indices = np.random.randint(0, dataset[0][0].shape[1])
    for l in range(kwargs['nEpochs']):
        y1 = Graph @ y
        y2 = compute_grad(y, inner_loss, dataset[0], indices, device)
        y = y1 - 10*kwargs['alpha'] * y2 
        acc.append(criterion(y, dataset[1][0], dataset[1][1].long(), device).item())
    torch.cuda.empty_cache()
    return y, acc

def DSGD(dataset, Graph, inner_loss, criterion, **kwargs):
    if "device" in kwargs.keys():
        device = kwargs["device"]
    else:
        device = "cuda"

    # Random Initialization
    y = torch.distributions.Normal(0.0, 5).sample((kwargs['nAgents'], kwargs['LLSize'])).float().to(device)
    
    # Forward path
    acc = []
    acc.append(accuracy(y, dataset[1][0], dataset[1][1].long(), device).item())
    indices = np.random.randint(0, dataset[0][0].shape[1])
    for l in range(kwargs['nEpochs']):
        y1 = Graph @ y
        y2 = compute_grad(y, inner_loss, dataset[0], indices, device)
        y = y1 - 100* kwargs['alpha'] * y2 
        acc.append(criterion(y, dataset[1][0], dataset[1][1].long(), device).item())
    torch.cuda.empty_cache()
    return y, acc

def DFedAvg(dataset, Graph, inner_loss, criterion, **kwargs):
    if "device" in kwargs.keys():
        device = kwargs["device"]
    else:
        device = "cuda"

    # Random Initialization
    y = torch.distributions.Normal(0.0, 5).sample((kwargs['nAgents'], kwargs['LLSize'])).float().to(device)
    
    # Forward path
    acc = []
    acc.append(accuracy(y, dataset[1][0], dataset[1][1].long(), device).item())
    indices =  np.random.randint(0, dataset[0][0].shape[1], 10)
    for l in range(kwargs['nEpochs']):
        y1 = Graph @ y
        y2 = compute_grad(y, inner_loss, dataset[0], indices, device)
        y = y1 - kwargs['alpha'] * y2 
        oldy1 = y.clone()
        oldy2 = y.clone()
        for k in range(5):
            dy = compute_grad(y, inner_loss, dataset[0], indices, device)
            y = y - 1 * dy + 0.1 * (oldy1 - oldy2)
            oldy1 = y.clone()
            oldy2 = oldy1.clone()
        acc.append(criterion(y, dataset[1][0], dataset[1][1].long(), device).item())
    torch.cuda.empty_cache()
    return y, acc

def metric(metadataset, Graph, inner_loss, criterion=accuracy, **kwargs):
    device = kwargs["device"]
    testAccuracy = np.zeros((len(metadataset), kwargs['nEpochs']+1))
    testAccuracy_DSGD = np.zeros((len(metadataset), kwargs['nEpochs']+1))
    testAccuracy_DFedAvg = np.zeros((len(metadataset), kwargs['nEpochs']+1))
    for ibatch in tqdm(range(len(metadataset))):
        iGraph = torch.tensor(Graph[ibatch], device=device).float()
        _, acc = DGD(metadataset[ibatch], iGraph, inner_loss, criterion, **kwargs)
        _, acc2 = DSGD(metadataset[ibatch], iGraph, inner_loss, criterion, **kwargs)
        _, acc3 = DFedAvg(metadataset[ibatch], iGraph, inner_loss, criterion, **kwargs)
        testAccuracy[ibatch] = acc
        testAccuracy_DSGD[ibatch] = acc2
        testAccuracy_DFedAvg[ibatch] = acc3
    return testAccuracy, testAccuracy_DSGD, testAccuracy_DFedAvg

    
