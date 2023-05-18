import numpy as np
import torch
import torch.nn as nn 

#%% Loss and Accuracy
def inner_loss(last_layer, images, targets, criterion=nn.CrossEntropyLoss()):
    loss = 0
    if len(last_layer.shape) < len(images.shape):
        last_layer = torch.reshape(last_layer, (last_layer.shape[0], images.shape[-1]+1, -1))
    logits =  (images @ last_layer[:,:-1] + last_layer[:, -1, None]).cpu()
    outputs = torch.max(logits, axis=2)[1]
    for i in range(outputs.shape[0]):
        loss += criterion(logits[i], torch.argmax(targets[i], dim=1))
    return loss/targets.numel()

def Lagrang_loss(last_layer, test_dataset, objective_function, criterion, penalty, device):
    loss = objective_function(last_layer, test_dataset[0].float().to(device), test_dataset[1].long(), criterion) + penalty
    return loss


# %% Computing the constraints
# def AsynPenalty(Labels, outsTrain, device):
#     """
#     assess whether the distance to the optimal decreases over the layers and
#     ensure that the updated position is far away from the previous one.
#     It helps to mitigate the asynchronous effect.
#     """
#     eps = 0.9
#     nExamples = Labels.shape[0]
#     N = Labels.shape[1]
#     p = outsTrain[0].shape[2]
#     L = len(outsTrain.keys()) - 1

#     cons1 = compute_penalty2(Labels, outsTrain, device)
#     cons2 = torch.zeros(nExamples, L, N).to(device)
#     X = torch.zeros(nExamples, L+1, N, p).to(device)
#     for l in outsTrain.keys():
#         X[:,l] = outsTrain[l] - Labels
#     cons2 = torch.norm(X[:, 0:-1], p=2, dim=3) - torch.norm(X[:, 1:], p=2, dim=3) - eps * torch.unsqueeze(torch.norm(Labels, p=2, dim=2), 1)
#     cons = torch.cat((cons1, cons2), dim=1)
#     return cons

# %% Computing the constraints
def compute_grad(z, objective_function, test_dataset, indices, device):
    z.requires_grad_()
    images = test_dataset[0][:,indices].float().to(device)
    images.requires_grad_()
    test_labels = test_dataset[1][:,indices].float()
    test_labels.requires_grad_()
    loss = objective_function(z, images, test_labels)
    return torch.autograd.grad(loss, z)[0]

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
