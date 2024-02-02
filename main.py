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
# Logging and Saving Path
FLAGS = argparse.ArgumentParser()
_, args = utils.parser(FLAGS)
modelPath = utils.Logging_Saving(args)

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
logging.info("MetaDataset Created ...")

# Which problenm to solve
featureSizePerClass = CNN.module.linear.in_features
LLSize = (featureSizePerClass+1)*args.nClasses
objective_function = inner_loss


# Create Graphs
if args.GraphType == 'Kdegree':
    Graph, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
elif args.GraphType == 'Random':
    Graph = utils.Generate_randGraphs(args.nDatasets, args.nAgents, 0.1)
elif args.GraphType == 'star':
    Graph = utils.Generate_starGraphs(args.nDatasets, args.nAgents)
logging.info("Graphs Created ...")

# Initialize/Load the unrolled model # Use UnrolledDGD_classical instead for classical FL
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, 
                        LLSize, args.batchSize, repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
#model = nn.DataParallel(model)
model = model.to(device)
logging.info("Unrolled Model Created ...")                
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
logging.info("Evaluation ...")
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
if args.GraphType == 'Kdegree':
    GraphTest, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
elif args.GraphType == 'Random':
    GraphTest = utils.Generate_randGraphs(args.nDatasets, args.nAgents, 0.1)
elif args.GraphType == 'star':
    GraphTest = utils.Generate_starGraphs(args.nDatasets, args.nAgents)

# Load best model
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, LLSize, args.batchSize,
                        repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
logging.info("Evaluation ...")
checkpoint = torch.load(modelPath+"model_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

testloss, testAccuracy, dist2Opt, _, _, _ = utils_test.metrics(model, test_metadataset, GraphTest, objective_function, criterion)


# %% Centralized CNN
logging.info("="*60)
logging.info("="*60)
logging.info('CNN Experiments')
centralized_training(test_metadataset, criterion, args)
logging.info("="*60)
logging.info("="*60)


print("OK!")
