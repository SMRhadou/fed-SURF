import numpy as np
import logging
from tqdm import tqdm

import torch
import torch.nn as nn

from allObjectives import *
from evaluation import evaluate


# %% Training 
def train_uDGD(model, dataset, optimizer, objective_function, criterion, Graph, args, device, split, **kwargs):
    nSets = int(len(dataset) * split)
    modelPath = kwargs["modelPath"]
    lr_dual = args.lr_dual
    constrained = args.constrained
    noisyOuts = args.noisyOuts
    NU = np.zeros((args.nEpochs+1, model.nLayers))

    nu = torch.zeros((model.nLayers,), device=torch.device(device)).double()
    best = np.inf
    for epoch in tqdm(range(args.nEpochs)):
        # Training
        model.train()
        if constrained:
            model, nu = constrained_learning(model, dataset, optimizer, objective_function, criterion, nu, SysID=Graph, 
                                            nSets=nSets, noisyOuts=noisyOuts, nClasses = kwargs['nClasses'], 
                                            lr_dual = lr_dual, eps=args.eps, device=device)
            NU[epoch+1] = nu.detach().cpu().numpy()
            logging.debug('duals {}'.format(list(nu.detach().cpu().numpy())))
        else:
            model = unconstrained_learning(model, dataset, optimizer, objective_function, criterion, SysID=Graph, 
                                            nSets=nSets, noisyOuts=noisyOuts, nClasses = kwargs['nClasses'], eps=args.eps, device=device)

        # Validation
        k = 0
        validloss = 0
        validAccuracy = 0
        model.eval()
        for iset in range(nSets, len(dataset)):
            printing = True if iset%20==0 else False
            loss, accuracy, _ = evaluate(model, dataset[iset], objective_function, criterion,  eps=args.eps, printing=printing, SysID=Graph[iset], device=device, epoch=epoch, iset=iset)
            validloss += loss
            validAccuracy += accuracy
            k += 1
        validloss /= k
        validAccuracy /= k

        # Save model
        if validloss < best:
            best = validloss
            bestAccuracy = validAccuracy
            epochBest = epoch
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "valid_loss": validloss
                        }, modelPath+"model_best.pth")
        logging.debug("validloss {} ({:.2f}), best: {} ({:.2f}) achieved at epoch {}\n".format(validloss, validAccuracy, best, bestAccuracy, epochBest))
            
        del validloss
        torch.cuda.empty_cache()
    return model, NU

# %% Learning Routines
def constrained_learning(model, dataset, optimizer, objective_function, criterion, nu, **kwargs):
    device = kwargs["device"]
    lr_dual = kwargs["lr_dual"]
    Graph = kwargs["SysID"]
    nSets = kwargs["nSets"]

    for iset in range(nSets):
        # Forward Step 
        images, labels = dataset[iset][0]
        S = torch.tensor(Graph[iset], device=device).float()
        last_layer, outs, indices = model(images, labels, S, **kwargs)
        cons, _ = GradPenalty(objective_function, dataset[iset][0], outs, indices, **kwargs)
        # Lagrangian Function
        penalty = torch.sum(nu * cons)
        Lagrang = Lagrang_loss(last_layer, dataset[iset][1], objective_function, criterion, penalty, device)
        # Primal update
        Lagrang.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.empty_cache()
    # for iset in range(nSets):
    #     with torch.no_grad():
    #         images, labels = dataset[iset][0]
    #         #images = images.float().to(device)
    #         #labels = labels.float().to(device)
    #         S = torch.tensor(Graph[iset], device=device).float()
    #         last_layer, outs, indices, = model(images, labels, S, **kwargs)
    #     temp, _ = GradPenalty(objective_function, dataset[iset][0], outs, indices, **kwargs)
    #     cons += temp
    # cons.div(nSets)
        # Dual update
        nu_temp = nu + lr_dual * cons
        nu = nn.ReLU()(nu_temp)
        nu = nu.detach()

    del images, cons
    torch.cuda.empty_cache()
    return model, nu


def unconstrained_learning(model, dataset, optimizer, objective_function, criterion, **kwargs):
    device = kwargs["device"]
    Graph = kwargs["SysID"]
    nSets = kwargs["nSets"]
    
    loss = 0
    for iset in range(nSets):
        images, labels = dataset[iset][0]
        S = torch.tensor(Graph[iset], device=device).float()
        last_layer, _, _ = model(images, labels, S, noisyOutputs=False)
        loss = objective_function(last_layer, dataset[iset][1][0].float().to(device), dataset[iset][1][1].long(), criterion)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    return model

