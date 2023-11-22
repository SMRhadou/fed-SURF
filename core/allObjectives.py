import numpy as np
import torch
import torch.nn as nn 
from torch.autograd.functional import jacobian

#%% Loss and Accuracy
def inner_loss(last_layer, images, targets, criterion=nn.CrossEntropyLoss()):
    loss = 0
    if len(images.shape) < 3:
        images = images.unsqueeze(1)
        targets = targets.unsqueeze(1)
    if len(last_layer.shape) < len(images.shape):
        last_layer = torch.reshape(last_layer, (last_layer.shape[0], images.shape[-1]+1, -1))
    logits =  (images @ last_layer[:,:-1] + last_layer[:, -1, None]).cpu()
    outputs = torch.max(logits, axis=2)[1]
    for i in range(outputs.shape[0]):
        loss += criterion(logits[i], torch.argmax(targets[i], dim=1))
    return loss/outputs.numel()

def Lagrang_loss(last_layer, test_dataset, objective_function, criterion, penalty, device):
    loss = objective_function(last_layer, test_dataset[0].float().to(device), test_dataset[1].long(), criterion) + penalty
    return loss

# %% Computing the constraints
def compute_grad(z, objective_function, test_dataset, indices, device):
    # z.requires_grad_()
    images = test_dataset[0][:,indices].float().to(device)
    # images.requires_grad_()
    test_labels = test_dataset[1][:,indices].float()
    # test_labels.requires_grad_()
    return  jacobian(objective_function, (z, images, test_labels), create_graph=True)[0]
    # loss = objective_function(z, images, test_labels)
    # return torch.autograd.grad(loss, z)[0]

def GradPenalty(objective_function, dataset, outsTrain, indices, **kwargs):
    """
    evaluate Supermartingale constraints
    """
    eps = kwargs['eps']
    torch.cuda.empty_cache()
    L = len(outsTrain.keys()) - 1
    gradVector = torch.zeros(L+1, outsTrain[0].shape[0], outsTrain[0].shape[1]).to(kwargs['device'])
    for l in outsTrain.keys():
        z = outsTrain[l]
        if l < L:
            gradVector[l] = compute_grad(z, objective_function, dataset, indices[l], kwargs['device'])
        else:
            idx = np.random.randint(0, dataset[0].shape[1], indices[0].shape[0])
            gradVector[l] = compute_grad(z, objective_function, dataset, idx, kwargs['device'])
    cons = (torch.norm(gradVector[1:], p='fro', dim=(1,2)) / torch.norm(gradVector[:-1], p='fro', dim=(1,2))) - (1-eps)
    return cons, torch.norm(gradVector, p='fro', dim=(1,2))     #L+1 x nAgents
