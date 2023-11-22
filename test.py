import numpy as np
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle 
import argparse
import torch

import core.utils_unrolling as utils
import core.utils_testing as utils_test
from models.unrolledModels import *
from models.ResNet import *
from models.baselines import *
import models.DGD as DGD
from core.evaluation import *
from core.training import *
from core.data import *


plt.set_loglevel (level = 'warning')
pil_logger = logging.getLogger('PIL')  
pil_logger.setLevel(logging.INFO)

np.random.seed(0)
torch.manual_seed(0)
FLAGS = argparse.ArgumentParser()
_, args = utils.parser(FLAGS)

logfile = f"./logs/logfile_MoreExp.log"
logging.basicConfig(filename=logfile, level=logging.DEBUG)
utils.printing(vars(args))

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the cnn model
test_dataset = loadDataset(False, args.Dataset)
CNN = utils.loadCNN(ResNetConv18, args.Dataset, device)
featureSizePerClass = CNN.module.linear.in_features

# Create Test Meta Dataset
if args.createMetaDataset:
    metadataset = createMetaDirichletDataset(CNN, test_dataset, args, test=True)
    if not os.path.exists("data/meta"):
        os.makedirs("data/meta")
    with open(f"./data/meta/{args.Dataset}-test-dirichlet-{args.alpha}.pkl", 'wb') as ObjFile:
        pickle.dump(metadataset, ObjFile)
else:
    with open(f"./data/meta/{args.Dataset}-test-dirichlet-{args.alpha}.pkl", 'rb') as ObjFile:
        metadataset = pickle.load(ObjFile)

# Generate Graphs
logging.debug("Generating Graphs ...")
Graph, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
# Graph = utils.Generate_randGraphs(args.nDatasets, args.nAgents, 0.1)

criterion = nn.CrossEntropyLoss()
objective_function = inner_loss

#%% Evaluation

# Load UDGD model (constrained)
modelPath = "./savedModels/CIFAR10_constrained_final_10_10/"
model = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, (featureSizePerClass+1)*args.nClasses, args.batchSize)
logging.debug("Evaluation ...")
checkpoint = torch.load(modelPath+"model_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

testloss, testAccuracy, dist2Opt, AccPerAgent, crossLoss, _ = utils_test.metrics(model, metadataset, Graph, objective_function, criterion)

# Load UDGD model (unconstrained)
modelPath = "./savedModels/CIFAR10_unconstrained_final_10_10/"
model_unconstrained = UnrolledDGD(args.nLayers, args.K, (featureSizePerClass+args.nClasses)*args.batchSize, (featureSizePerClass+1)*args.nClasses, args.batchSize)
logging.debug("Evaluation ...")
checkpoint = torch.load(modelPath+"model_best.pth")
model_unconstrained.load_state_dict(checkpoint["model_state_dict"])

testloss_unconstrained, testAccuracy_unconstrained, dist2Opt_unconstrained, _, _, _  = utils_test.metrics(model_unconstrained, metadataset, Graph, objective_function, criterion)

# Plotting
utils_test.plotting(testloss, testAccuracy, dist2Opt, testloss_unconstrained, testAccuracy_unconstrained, dist2Opt_unconstrained, title='performance_test')


# Decentralized Benchmarks
# Graph, _ = utils.Generate_KdegreeGraphs(args.nDatasets, args.nAgents, args.nodeDegree)
testAccuracy_DGD, testAccuracy_DSGD, testAccuracy_DFedAvg = DGD.metric(metadataset, Graph, inner_loss, criterion=accuracy, nEpochs=200, alpha=100, nAgents=args.nAgents, LLSize=(featureSizePerClass+1)*args.nClasses, device=device)
utils_test.plotting_decentralizedFigs(testAccuracy, testAccuracy_DGD, testAccuracy_DSGD, testAccuracy_DFedAvg, args, title='decen_performance_0.7')
with open(f"./data/Accuracy-{args.alpha}.pkl", 'wb') as ObjFile:
    pickle.dump((testAccuracy, testAccuracy_DGD, testAccuracy_DSGD, testAccuracy_DFedAvg),ObjFile)

# %% Centralized CNN
logging.debug("="*60)
logging.debug("="*60)
logging.debug('CNN Experiments')
centralized_training(metadataset, criterion, args)
logging.debug("="*60)
logging.debug("="*60)

#%% Asynchronous Communications
pickDataset = 0
constrainedAsyn = []
unconstrainedAsyn = []
constrainedAsynAcc = []
unconstrainedAsynAcc = []
for i in tqdm(np.arange(11)):
    print(i)
    loss_cons = []
    loss_uncons = []
    acc_cons = []
    acc_uncons = []
    for j in np.arange(100):
        testloss, testAccuracy, _, _, _, BOagents = utils_test.metrics(model, (metadataset[pickDataset],), 
                                                                    Graph, objective_function, criterion, evaluate=evaluateAsyn, nBOagents=i)

        testloss_unconstrained, testAccuracy_unconstrained, _, _, _, BOagents = utils_test.metrics(model_unconstrained, (metadataset[pickDataset],), 
                                                                    Graph, objective_function, criterion, evaluate=evaluateAsyn, nBOagents=i, BOagents=BOagents)

        loss_cons.append(testloss[:,-1])
        loss_uncons.append(testloss_unconstrained[:,-1])
        acc_cons.append(testAccuracy[:,-1])
        acc_uncons.append(testAccuracy_unconstrained[:,-1])
        
    constrainedAsyn.append((np.mean(loss_cons), np.std(loss_cons)))
    constrainedAsynAcc.append((np.mean(acc_cons), np.std(acc_cons)))
    unconstrainedAsyn.append((np.mean(loss_uncons), np.std(loss_uncons)))
    unconstrainedAsynAcc.append((np.mean(acc_uncons), np.std(acc_uncons)))
    logging.debug('Trial {} - Constrained: {} +/- {}, Unconstrained: {} +/- {}'.format(i, np.mean(loss_cons), np.std(loss_cons), np.mean(loss_uncons), np.std(loss_uncons)))
    logging.debug('Trial {} - Constrained: {} +/- {}, Unconstrained: {} +/- {}'.format(i, np.mean(acc_cons), np.std(acc_cons), np.mean(acc_uncons), np.std(acc_uncons)))
    # Plotting
utils_test.plotAsyn(constrainedAsyn, unconstrainedAsyn, title='Loss')
utils_test.plotAsyn(constrainedAsynAcc, unconstrainedAsynAcc, title='Accuracy')

print('OK')

