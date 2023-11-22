import numpy as np
import logging 
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



from core.evaluation import *
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"


def metrics(model, metadataset, Graph, objective_function, criterion, evaluate=evaluate, **kwargs):
    if 'nBOagents' in kwargs.keys():
        nBOagents = kwargs['nBOagents']
    else:
        nBOagents = None
    if "BOagents" in kwargs.keys():
        BOagents = kwargs["BOagents"]
    else:
        BOagents = None
    device = 'cpu' 
    model = model.to(device)
    model.eval()
    k = 0
    testloss =  np.zeros((len(metadataset), model.nLayers+1))
    testAccuracy = np.zeros((len(metadataset), model.nLayers+1))
    dist2Opt = np.zeros((len(metadataset), model.nLayers+1))
    Acc = []
    obj = []
    AccPerAgent = np.zeros((len(metadataset), ))
    model.eval()
    crossLoss = torch.zeros(len(metadataset), len(metadataset))
    for ibatch in tqdm(range(len(metadataset))):
        loss, acc, outs = evaluate(model, metadataset[ibatch], objective_function, criterion,  eps=0.05, nBOagents=nBOagents, SysID=Graph[ibatch], device=device, ibatch=ibatch, BOagents=BOagents)
        testloss[ibatch] = loss
        testAccuracy[ibatch] = acc
        if type(outs) is tuple:
            BOagents = outs[1]
            outs = outs[0]
        dist2Opt[ibatch] = np.array([torch.norm(outs[l] - outs[model.nLayers], p=2, dim=(0,1)).item() for l in range(model.nLayers+1)])
        obj.append(loss[-1].item())
        Acc.append(acc[-1].item())
        AccPerAgent[ibatch] = accuracyPerAgents(outs[model.nLayers], metadataset[ibatch][1][0], metadataset[ibatch][1][1], device)[1]#.append(accuracyPerAgents(outs[model.nLayers], metadataset[ibatch][1][0], metadataset[ibatch][1][1], device))
        k += 1
        with torch.no_grad():
            for i in range(len(metadataset)):
                crossLoss[ibatch, i] = accuracy(outs[model.nLayers], metadataset[i][1][0], metadataset[i][1][1], device)

    logging.debug(r'Accuracy {:.2f} +/- {:.2f}'.format(np.mean(Acc)*100, np.std(Acc)*100))
    logging.debug(r'Objective {} +/- {}'.format(np.mean(obj), np.std(obj)))
    logging.debug(r'Variability Per Agent {:.2f} +/- {:.2f}'.format(np.mean(AccPerAgent)*100, np.std(AccPerAgent)*100))
    return testloss, testAccuracy, dist2Opt, AccPerAgent, crossLoss, BOagents

def plotting(loss_constrained, acc, dist2Opt, loss_unconstrained, acc_unconstrained, dist2Opt_unconstrained, title):
    dist2Opt[len(dist2Opt)-1] = 10
    dist2Opt_unconstrained[len(dist2Opt_unconstrained)-1] = 10
    if not os.path.exists("figs"):
        os.makedirs("figs")
    # Figure 1: Distance to optimal
    sns.set_context('notebook')
    sns.set_style('darkgrid')
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(np.arange(loss_constrained.shape[1]), np.mean(loss_constrained, axis=0), 'b', label='SURF')
    plt.errorbar(np.arange(loss_constrained.shape[1]), np.mean(loss_constrained, axis=0), yerr=np.std(loss_constrained, axis=0), fmt='b', capsize=3, alpha=0.5)
    plt.plot(np.arange(loss_unconstrained.shape[1]), np.mean(loss_unconstrained, axis=0), 'r', label='DGD')
    plt.errorbar(np.arange(loss_unconstrained.shape[1]), np.mean(loss_unconstrained, axis=0), yerr=np.std(loss_unconstrained, axis=0), fmt='r', capsize=3, alpha=0.5)
    plt.yscale('log')
    plt.xlabel("layer $l$")
    plt.ylabel("loss $f(\mathbf{W}_l)$")
    plt.legend()

    plt.subplot(1,2,2)
    K = 2
    plt.plot(np.arange(acc.shape[1]), np.mean(acc, axis=0)*100, 'b', label='U-DGD')
    plt.errorbar(np.arange(acc.shape[1]), np.mean(acc, axis=0)*100, yerr=np.std(acc, axis=0)*100, fmt='b', capsize=3, alpha=0.5)
    plt.plot(np.arange(acc_unconstrained.shape[1]), np.mean(acc_unconstrained, axis=0)*100, 'r', label='DGD')
    plt.errorbar(np.arange(acc_unconstrained.shape[1]), np.mean(acc_unconstrained, axis=0)*100, yerr=np.std(acc_unconstrained, axis=0)*100, fmt='r', capsize=3, alpha=0.5)
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/{title}.pdf')
    sns.reset_orig()


def plotting_decentralizedFigs(acc, acc2, acc3, acc4, args, title):
    if not os.path.exists("figs"):
        os.makedirs("figs")
    # Figure 1: Distance to optimal
    sns.set_context('notebook')
    sns.set_style('darkgrid')
    plt.figure(figsize=(4,3))
    K = args.K - 1
    plt.plot(np.arange(0, K*acc.shape[1], K), np.mean(acc[:-1], axis=0)*100, 'b', label='U-DGD')
    plt.errorbar(np.arange(0, K*acc.shape[1], K), np.mean(acc[:-1], axis=0)*100, yerr=np.std(acc[:-1], axis=0)*100, fmt='b', capsize=3, alpha=0.5)
    plt.plot(np.arange(acc2.shape[1]), np.mean(acc2, axis=0)*100, 'r', label='DGD')
    plt.errorbar(np.arange(0, acc2.shape[1], 10), np.mean(acc2[:,::10], axis=0)*100, yerr=np.std(acc2[:,::10], axis=0)*100, fmt='r', capsize=3, alpha=0.5)
    plt.plot(np.arange(acc3.shape[1]), np.mean(acc3, axis=0)*100, 'k', label='DSGD')
    # plt.errorbar(np.arange(0, acc3.shape[1], 10), np.mean(acc3[:,::10], axis=0)*100, yerr=np.std(acc3[:,::10], axis=0)*100, fmt='r', capsize=3, alpha=0.5)
    plt.plot(np.arange(acc4.shape[1]), np.mean(acc4, axis=0)*100, 'gray', label='DFedAvgM')
    plt.errorbar(np.arange(0, acc4.shape[1], 10), np.mean(acc4[:,::10], axis=0)*100, yerr=np.std(acc4[:,::10], axis=0)*100, fmt='gray', capsize=3, alpha=0.5)
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/{title}.pdf')
    sns.reset_orig()

def plotAsyn(constrainedAsyn, unconstrainedAsyn, title):
    mean1 = [constrainedAsyn[i][0] for i in range(len(constrainedAsyn))]
    std1 = [constrainedAsyn[i][1] for i in range(len(constrainedAsyn))]
    mean2 = [unconstrainedAsyn[i][0] for i in range(len(unconstrainedAsyn))]
    std2 = [unconstrainedAsyn[i][1] for i in range(len(unconstrainedAsyn))]
    # Figure 1: Distance to optimal
    sns.set_context('notebook')
    sns.set_style('darkgrid')
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(len(constrainedAsyn)), mean1, 'b', label='SURF')
    plt.errorbar(np.arange(len(constrainedAsyn)), mean1, yerr=std1, fmt='b', capsize=3, alpha=0.5)
    plt.plot(np.arange(len(constrainedAsyn)), mean2, 'r', label='standard unrolling')
    plt.errorbar(np.arange(len(constrainedAsyn)), mean2, yerr=std2, fmt='r', capsize=3, alpha=0.5)
    plt.xlabel(r"$n_{asyn}$")
    plt.ylabel(f"{title}")
    if title == 'lossAsyn':
        plt.ylim(top=0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/{title}_Aysn.pdf')
    sns.reset_orig()
    