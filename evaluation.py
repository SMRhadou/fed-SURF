import torch
import logging
import pickle
import numpy as np

from allObjectives import GradPenalty

def accuracy(last_layer, images, targets, device):
    acc = 0
    if len(last_layer.shape) < len(images.shape):
        last_layer = torch.reshape(last_layer, (last_layer.shape[0], images.shape[-1]+1, -1))
    logits =  (images.float().to(device) @ last_layer[:,:-1] + last_layer[:, -1, None]).cpu()
    outputs = torch.max(logits, axis=2)[1]
    for i in range(outputs.shape[0]):
        acc += torch.sum(outputs[i] == torch.argmax(targets[i], dim=1))
    return acc/outputs.numel()

def accuracyPerAgents(last_layer, images, targets, device):
    acc = 0
    if len(last_layer.shape) < len(images.shape):
        last_layer = torch.reshape(last_layer, (last_layer.shape[0], images.shape[-1]+1, -1))
    logits =  (images.float().to(device) @ last_layer[:,:-1] + last_layer[:, -1, None]).cpu()
    outputs = torch.max(logits, axis=2)[1]
    acc = []
    for i in range(outputs.shape[0]):
        acc.append(torch.sum(outputs[i] == torch.argmax(targets[i], dim=1)).item()/outputs.shape[1])
    return np.mean(acc), np.std(acc)

def evaluate(model, dataset, objective_function, criterion, eps, printing=True, saveResults=False, resultPath=None, grad=False, **kwargs):
    device = kwargs["device"]
    Graph = kwargs["SysID"]
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
    else:
        epoch = 'Test'
    if 'iset' in kwargs:
        iset = kwargs['iset']
    else:
        iset = 0

    with torch.no_grad():
        images, labels = dataset[0]
        images = images.float().to(device)
        labels = labels.float().to(device)
        Graph = torch.tensor(Graph, device=device).float()
        last_layer, outsValid, indices = model(images, labels, Graph, noisyOutputs=True, device=device)
        if epoch == 'Test':
            validloss = []
            validAccuracy = []
            for i in range(model.nLayers+1):
                validloss.append(objective_function(outsValid[i], dataset[1][0].float().to(device), dataset[1][1].long(), criterion))
                validAccuracy.append(accuracy(outsValid[i], dataset[1][0], dataset[1][1].long(), device))
        else:
            validloss = objective_function(last_layer, dataset[1][0].float().to(device), dataset[1][1].long(), criterion)
            validAccuracy = accuracy(last_layer, dataset[1][0], dataset[1][1].long(), device)
    
    if printing and epoch != 'Test':
        cons, grad = GradPenalty(objective_function, dataset[0], outsValid, indices, eps=eps, **kwargs)
        logging.debug('gradients {}'.format(list(grad.detach().cpu().numpy())))
        logging.debug('constraints {}'.format(list(cons.detach().cpu().numpy())))
        logging.debug('loss across layers: {}'.format([objective_function(outsValid[i], dataset[1][0].float().to(device),
                                                            dataset[1][1].long(), criterion).item() for i in range(model.nLayers+1)]))
        
        logging.debug("Epoch {}, batch {}, Loss {:.4f}, accuracy {:.2f}".format(epoch, iset, validloss.item(), validAccuracy.item()))
        
    # Save results
    if saveResults and resultPath is not None:
        with open(resultPath, 'wb') as ObjFile:
            pickle.dump((Graph, images, outsValid, kwargs['NU'], eps), ObjFile)
        
    return validloss, validAccuracy, outsValid


def evaluateAsyn(model, dataset, objective_function, criterion, eps, nBOagents=1, printing=True, saveResults=False, resultPath=None, grad=False, **kwargs):
    device = kwargs["device"]
    Graph = kwargs["SysID"]
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
    else:
        epoch = 'Test'
    if 'iset' in kwargs:
        iset = kwargs['iset']
    else:
        iset = 0

    with torch.no_grad():
        images, labels = dataset[0]
        images = images.float().to(device)
        labels = labels.float().to(device)
        Graph = torch.tensor(Graph, device=device).float()
        last_layer, outsValid, indices = model.forwardAsyn(images, labels, Graph, nBOagents, device=device)
        if epoch == 'Test':
            validloss = []
            validAccuracy = []
            for i in range(model.nLayers+1):
                validloss.append(objective_function(outsValid[i], dataset[1][0].float().to(device), dataset[1][1].long(), criterion))
                validAccuracy.append(accuracy(outsValid[i], dataset[1][0], dataset[1][1].long(), device))
        else:
            validloss = objective_function(last_layer, dataset[1][0].float().to(device), dataset[1][1].long(), criterion)
            validAccuracy = accuracy(last_layer, dataset[1][0], dataset[1][1].long(), device)
    
    if printing and epoch != 'Test':
        cons, grad = GradPenalty(objective_function, dataset[0], outsValid, indices, eps=eps, **kwargs)
        logging.debug('gradients {}'.format(list(grad.detach().cpu().numpy())))
        logging.debug('constraints {}'.format(list(cons.detach().cpu().numpy())))
        logging.debug('loss across layers: {}'.format([objective_function(outsValid[i], dataset[1][0].float().to(device),
                                                            dataset[1][1].long(), criterion).item() for i in range(model.nLayers+1)]))
        
        logging.debug("Epoch {}, batch {}, Loss {:.4f}, accuracy {:.2f}".format(epoch, iset, validloss.item(), validAccuracy.item()))
        
    # Save results
    if saveResults and resultPath is not None:
        with open(resultPath, 'wb') as ObjFile:
            pickle.dump((Graph, images, outsValid, kwargs['NU'], eps), ObjFile)
        
    return validloss, validAccuracy, outsValid