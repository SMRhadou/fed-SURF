import numpy as np
from tqdm import tqdm
import logging 
import argparse
import os
import pickle

import torch
import torch.backends.cudnn as cudnn


def parser(FLAGS):
    FLAGS = argparse.ArgumentParser(description='SURF')
    FLAGS.add_argument('--Trial', type=str, default='Dirichlet', help='Trial')
    FLAGS.add_argument('--Dataset', type=str, default='CIFAR10', help='Dataset')
    # Model Parameters
    FLAGS.add_argument('--nLayers', type=int, default=10, help='nLayers')
    FLAGS.add_argument('--coreLayers', type=int, default=5, help='Core layers to be repeated')
    FLAGS.add_argument('--K', type=int, default=3, help='FilterTaps')
    # Learning Parameters
    FLAGS.add_argument('--nEpochs', type=int, default=200, help='nEpochs')
    FLAGS.add_argument('--lr', type=float, default=1e-3, help='lr')
    FLAGS.add_argument('--lr_dual', type=float, default=1e-2, help='lr_dual')
    FLAGS.add_argument('--eps', type=float, default=0.01, help='epsilon')
    # Data Parameters
    FLAGS.add_argument('--alpha', type=float, default=1, help='Dirichlet concentration parameter')
    FLAGS.add_argument('--nDatasets', type=int, default=600, help='Size of Meta Dataset')
    FLAGS.add_argument('--nAgents', type=int, default=100, help='nAgents')
    FLAGS.add_argument('--nodeDegree', type=int, default=3, help='Node degree')
    FLAGS.add_argument('--subDatasetSize', type=int, default=6000, help='Size of the subdatasets')
    FLAGS.add_argument('--nTrainPerAgent', type=int, default=45, help='number of examples per agent')
    FLAGS.add_argument('--nClasses', type=int, default=10, help='number of classes in the subdataset')
    FLAGS.add_argument('--batchSize', type=int, default=10, help='number of examples fed to an unrolled layer')
    # Features
    FLAGS.add_argument('--constrained', action="store_true")
    FLAGS.add_argument('--noisyOuts', action="store_true")
    FLAGS.add_argument('--createMetaDataset', action="store_true")
    FLAGS.add_argument('--repeatLayers', action="store_true")
    FLAGS.add_argument('--pretrained', action="store_true")
    FLAGS.add_argument('--gpuID', type=str, default="1", help='choose a GPU')
    FLAGS.add_argument('--mode', type=str, default="0", help='0-last layer of a resnet or 1-lstm')
    return FLAGS, FLAGS.parse_args()

def Generate_KdegreeGraphs(nExamples:int, N:int, K:int):
    Graph = np.zeros((nExamples, N, N))
    nGraph = np.zeros((nExamples, N, N))
    for exp in range(nExamples):
        S = np.eye(N)
        for k in range(1, int(K/2)+1):
            S += np.eye(N, k=k) + np.eye(N, k=-k) + np.eye(N,k=N-k) + np.eye(N,k=-N+k)
        if K%2 != 0:
            d = int(N/2) + 1
            D = np.eye(N, k=d)
            S += D + D.T
        P = np.random.permutation(np.eye(N))
        S = P @ S @ P.T
        Graph[exp] = (S.T/np.sum(S, axis=1)).T
        nGraph[exp] = S
    return Graph, nGraph

def Generate_starGraphs(nExamples:int, N:int):
    Graph = np.zeros((nExamples, N, N))
    nGraph = np.zeros((nExamples, N, N))
    for exp in range(nExamples):
        S = np.zeros((N, N))
        S[:,0] = 1
        S[0,:] = 1
        S[0,0] = 0
        S /= np.real(np.max(np.linalg.eigvals(S)))
        Graph[exp] = S
    return Graph

def Generate_randGraphs(nExamples:int, N:int, p:float):
    print("Generate new graphs ...")
    Graph = np.zeros((nExamples, N, N))
    for exp in tqdm(range(nExamples)):
        # Construct the graph
        S = (np.random.uniform(0, 1, (N,N)) < p).astype(int)
        S = ((S + S.T + np.eye(N)) >0).astype(float) 
        S /= np.real(np.max(np.linalg.eigvals(S)))
        Graph[exp] = S
    return Graph

def printing(args):
    logging.debug("="*60)
    for i, item in args.items():
        logging.debug("{}: {}".format(i, item))
    logging.debug("="*60)


def loadCNN(modelName, datasetName, device):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if datasetName == 'CIFAR10':
        checkpoint = torch.load('./checkpoint/ckpt18VHB.pth')
        inDim = 3
    elif datasetName == 'MNIST':
        checkpoint = torch.load('./checkpoint/ckptMNIST.pth')
        inDim = 1

    CNN = modelName(inDim)
    CNN.to(device)
    if device == 'cuda':
        CNN = torch.nn.DataParallel(CNN)
        cudnn.benchmark = True
    CNN.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return CNN

def Logging_Saving(args):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logfile = f"./logs/logfile_{args.Dataset}_{args.Trial}_{args.nLayers}_{args.nClasses}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG)

    if not os.path.exists(f"savedModels/{args.Dataset}_{args.Trial}_{args.nLayers}_{args.nClasses}"):
        os.makedirs(f"savedModels/{args.Dataset}_{args.Trial}_{args.nLayers}_{args.nClasses}")
    modelPath = f"./savedModels/{args.Dataset}_{args.Trial}_{args.nLayers}_{args.nClasses}/"

    with open(modelPath+"args.pkl", 'wb') as ObjFile:
        pickle.dump(args, ObjFile)
    return modelPath

