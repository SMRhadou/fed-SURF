import pickle
import os
import logging
import argparse
import random

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim

import core.utils_unrolling as utils
import core.utils_testing as utils_test
from models.unrolledModels import *
from models.baselines import *
from models.ResNet import *
from core.training import *
from core.evaluation import *
from core.data import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


# Logging and Saving Path
FLAGS = argparse.ArgumentParser()
_, args = utils.parser(FLAGS)
modelPath = utils.Logging_Saving(args)

# Cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CNN model
dataset = loadDataset(True, args.Dataset)
CNN = utils.loadCNN(ResNetConv18, args.Dataset, device)

# Create Meta Dataset
if args.createMetaDataset:
    metadataset = createMetaDirichletDataset(CNN, dataset, args)
    if not os.path.exists("data/meta"):
        os.makedirs("data/meta")
    with open(f"./data/meta/{args.Dataset}-Dirichlet-{args.alpha}.pkl", 'wb') as ObjFile:
        pickle.dump(metadataset, ObjFile)
else:
    with open(f"./data/meta/{args.Dataset}-Dirichlet-{args.alpha}.pkl", 'rb') as ObjFile:
        metadataset = pickle.load(ObjFile)
logging.debug("MetaDataset Created ...")

# Which problenm to solve
featureSizePerClass = CNN.module.linear.in_features
LLSize = (featureSizePerClass+1)*args.nClasses
objective_function = inner_loss


# Create Graphs
Graph, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
# Graph = utils.Generate_randGraphs(args.nDatasets, args.nAgents, 0.1)
logging.debug("Graphs Created ...")

# Initialize/Load the unrolled model
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, 
                        LLSize, args.batchSize, repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
#model = nn.DataParallel(model)
model = model.to(device)
logging.debug("Unrolled Model Created ...")                
if args.pretrained:
    checkpoint = torch.load(modelPath+"model_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

# Training
utils.printing(vars(args))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
_, _ = train_uDGD(model, metadataset, optimizer, objective_function, criterion, Graph, args, device, split=0.85, nClasses=args.nClasses, modelPath=modelPath)
utils.printing(vars(args))

# %% Evaluation
logging.debug("Evaluation ...")
# Create Test meta dataset
test_dataset = loadDataset(False, args.Dataset)
if args.createMetaDataset:
    test_metadataset = createMetaDataset(CNN, test_dataset, args, test=True)
    if not os.path.exists("data/meta"):
        os.makedirs("data/meta")
    with open(f"./data/meta/{args.Dataset}-test.pkl", 'wb') as ObjFile:
        pickle.dump(test_metadataset, ObjFile)
else:
    with open(f"./data/meta/{args.Dataset}-test.pkl", 'rb') as ObjFile:
        test_metadataset = pickle.load(ObjFile)

# Generate Graphs
GraphTest, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
# GraphTest = utils.Generate_randGraphs(args.nDatasets, args.nAgents, 0.1)

# Load best model
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, LLSize, args.batchSize,
                        repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
logging.debug("Evaluation ...")
checkpoint = torch.load(modelPath+"model_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

testloss, testAccuracy, dist2Opt, _, _, _ = utils_test.metrics(model, test_metadataset, GraphTest, objective_function, criterion)


# %% Centralized CNN
logging.debug("="*60)
logging.debug("="*60)
logging.debug('CNN Experiments')
centralized_training(test_metadataset, criterion, args)
logging.debug("="*60)
logging.debug("="*60)


print("OK!")
