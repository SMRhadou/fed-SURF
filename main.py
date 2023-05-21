import pickle
import os
import logging
import argparse

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim

import utils_unrolling as utils
import utils_testing as utils_test
from models.unrolledModels import *
from models.baselines import *
from models.ResNet import *
from training import *
from evaluation import *
from data import *


# Logging and Saving Path
FLAGS = argparse.ArgumentParser()
_, args = utils.parser(FLAGS)
modelPath = utils.Logging_Saving(args)

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CNN model
dataset = loadDataset(True)
CNN = utils.loadCNN(ResNetConv18, device)

# Data parameters
featureSizePerClass = CNN.module.linear.in_features

# Create Meta Dataset
if args.createMetaDataset:
    metadataset, classesDist = createMetaDataset(CNN, dataset, args)
    if not os.path.exists("data/meta"):
        os.makedirs("data/meta")
    with open(f"./data/meta/Experiment1_{args.nClasses}.pkl", 'wb') as ObjFile:
        pickle.dump((metadataset, classesDist), ObjFile)
else:
    with open(f"./data/meta/Experiment1_{args.nClasses}.pkl", 'rb') as ObjFile:
        metadataset, classesDist = pickle.load(ObjFile)
logging.debug("MetaDataset Created ...")

# Create Graphs
Graph, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
logging.debug("Graphs Created ...")

# Initialize/Load the unrolled model
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, 
                        (featureSizePerClass+1)*args.nClasses, args.batchSize,
                        repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
model = model.to(device)
logging.debug("Unrolled Model Created ...")                
if args.pretrained:
    checkpoint = torch.load(modelPath+"model_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

# Training
utils.printing(vars(args))
criterion = nn.CrossEntropyLoss()
objective_function = inner_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr)
_, _ = train_uDGD(model, metadataset, optimizer, objective_function, criterion, Graph, args, device, split=0.85, nClasses=args.nClasses, modelPath=modelPath)
utils.printing(vars(args))

# %% Evaluation
logging.debug("Evaluation ...")
# Create Test meta dataset
test_dataset = loadDataset(False)
if args.createMetaDataset:
    test_metadataset, _ = createMetaDataset(CNN, test_dataset, args, classesDist=classesDist, test=True)
    if not os.path.exists("data/meta"):
        os.makedirs("data/meta")
    with open(f"./data/meta/Experiment1-test_{args.nClasses}.pkl", 'wb') as ObjFile:
        pickle.dump((test_metadataset, classesDist), ObjFile)
else:
    with open(f"./data/meta/Experiment1-test_{args.nClasses}.pkl", 'rb') as ObjFile:
        test_metadataset, classesDist = pickle.load(ObjFile)

# Generate Graphs
GraphTest, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)

# Load best model
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, (featureSizePerClass+1)*args.nClasses, args.batchSize,
                        repeatLayers=args.repeatLayers, coreLayers=args.coreLayers)
logging.debug("Evaluation ...")
checkpoint = torch.load(modelPath+"model_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

testloss, testAccuracy, dist2Opt, _ = utils_test.metrics(model, test_metadataset, GraphTest, objective_function, criterion)


# %% Centralized CNN
logging.debug("="*60)
logging.debug("="*60)
logging.debug('CNN Experiments')
centralized_training(test_metadataset, criterion, args)
logging.debug("="*60)
logging.debug("="*60)


print("OK!")