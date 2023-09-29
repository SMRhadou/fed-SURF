import torch
import torch.nn as nn
import numpy as np

from core.training import compute_grad

class GraphFilter(nn.Module):
    def __init__(self, K: int):
        super(GraphFilter, self).__init__()
        self.K = K
        randw = torch.distributions.Uniform(0, 0.1).sample((K,))    # Initialize the filter taps
        self.weights = nn.Parameter(randw.float())
    
    def forward(self, X:torch.tensor, S:torch.tensor):
        kAggre = self.weights[0] * X.T
        xTilde = X.T
        for k in range(1, self.K):
            xTilde = xTilde @ S                     # Recursive application of S
            kAggre += self.weights[k] * xTilde
        return kAggre.T

class LinearLayer(nn.Module):
    def __init__(self, p0: int, p1: int):
        super(LinearLayer, self).__init__()
        self.p0 = p0
        self.p1 = p1
        randw = torch.distributions.Uniform(0, 0.1).sample((p0,p1))    # Initialize the filter taps
        self.weights = nn.Parameter(randw.float())
        randw = torch.distributions.Uniform(0, 0.1).sample((p1,))
        self.bias = nn.Parameter(randw.float())
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X:torch.tensor):
        return self.dropout(nn.BatchNorm1d(self.bias.shape[0], device=X.device)(X @ self.weights + self.bias))

class MLP(nn.Module):
    def __init__(self, width: list):
        super(MLP, self).__init__()
        self.width = width
        self.nLayers = len(width) - 1
        self.layers = nn.ModuleList()
        for l in range(self.nLayers):
            layer = LinearLayer(width[l],width[l+1])    # Initialize the filter taps
            self.layers.append(layer)
        self.activations = nn.ModuleDict({
            "tanh": nn.Tanh(),
            "ReLU": nn.ReLU()})

    def forward(self, X:torch.tensor):
        for l in range(self.nLayers):
            X = self.layers[l](X)
            X = nn.Tanh()(X)
        return X
    

class UnrolledDGD(nn.Module):
    def __init__(self, nLayers:int, K:int, dataSize:int, LLSize:int, batchSize:int, repeatLayers=False, coreLayers=10):
        super(UnrolledDGD, self).__init__()
        self.K = K
        self.nLayers = nLayers
        self.LLSize = LLSize
        self.dataSize = dataSize
        self.repeatLayers = repeatLayers
        self.batchSize = batchSize
        if self.repeatLayers:
            self.coreLayers = coreLayers
        else:
            self.coreLayers = nLayers
        self.layers = nn.ModuleList()
        for l in range(self.coreLayers):
            layer = nn.ModuleDict({
                "GF": GraphFilter(self.K),
                "linear":  LinearLayer(self.dataSize+self.LLSize, self.LLSize)})#MLP([self.dataSize+self.LLSize, 7000, self.LLSize])}) #MLP([self.dataSize+self.LLSize, 500, self.LLSize])})
            self.layers.append(layer)
        self.activations = nn.ModuleDict({
            "tanh": nn.Tanh(),
            "ReLU": nn.ReLU()})

    def forward(self, Features:torch.tensor, labels:torch.tensor, Graph:torch.tensor, noisyOuts=False, **kwargs):         
        nAgents = Features.shape[0]
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = "cuda"

        # Random Initialization
        y = torch.distributions.Normal(0.0, 5).sample((nAgents, self.LLSize)).float().to(device)
        
        # Forward path
        outs = {}
        outs[0] = y
        indices = []
        for l in range(self.nLayers):
            if self.repeatLayers:
                layer = self.layers[l%self.coreLayers]
            else:
                layer = self.layers[l]
            idx = np.random.randint(0, Features.shape[1], self.batchSize)
            indices.append(idx)
            data = torch.cat((Features[:,idx], labels[:,idx]), dim=2).reshape((Features.shape[0],-1)).float().to(device)
            y1 = layer["GF"](y, Graph)
            z = torch.cat((data, y), dim=1)
            y2 = layer["linear"](z)
            y2 = nn.ReLU()(y2)
            y = y1 - y2 
            # Noisy outputs
            if noisyOuts and l<self.nLayers-1:
                sigma = 1/(l+1)**2#torch.norm(compute_grad(x, kwargs['objective'], kwargs['test_dataset'], kwargs['device']), p=2, dim=1)
                y = y + torch.distributions.Normal(0.0, sigma).sample(y.shape).float().to(device)
            outs[l+1] = y
        torch.cuda.empty_cache()
        return y, outs, indices

    def forwardAsyn(self, Features, labels, Graph, nBOagents, BOflag=False, **kwargs):
        device = kwargs["device"]
        nAgents = Features.shape[0]
        if kwargs["BOagents"] is not None:
            BOagents = kwargs["BOagents"]
        else:
            BOflag = True
            BOagents = np.zeros((self.nLayers, nBOagents))

        # Random Initialization
        y = torch.distributions.Normal(0.0, 5).sample((nAgents, self.LLSize)).float().to(device)
        
        # Forward path
        outs = {}
        outs[0] = y
        indices = []
        for l in range(self.nLayers):
            if self.repeatLayers:
                layer = self.layers[l%self.coreLayers]
            else:
                layer = self.layers[l]
            if BOflag:
                BOagents[l] = np.random.choice(nAgents, nBOagents, replace=False)
            buffer = y[BOagents[l]]
            y[BOagents[l]] = 0.0
            idx = np.random.randint(0, Features.shape[1], self.batchSize)
            indices.append(idx)
            data = torch.cat((Features[:,idx], labels[:,idx]), dim=2).reshape((Features.shape[0],-1)).float().to(device)
            y1 = layer["GF"](y, Graph)
            z = torch.cat((data, y), dim=1)
            y2 = layer["linear"](z)
            y2 = nn.ReLU()(y2)
            y = y1 - y2 
            y[BOagents[l]]= buffer
            outs[l+1] = y
        torch.cuda.empty_cache()
        return y, outs, indices, BOagents

class UnrolledDGD_noGNN(UnrolledDGD):
    def __init__(self, nLayers:int, K:int, dataSize:int, LLSize:int, batchSize:int, repeatLayers=False, coreLayers=10):
        super(UnrolledDGD_noGNN, self).__init__(nLayers, K, dataSize, LLSize, batchSize, repeatLayers, coreLayers)
        self.layers = nn.ModuleList()
        for l in range(self.coreLayers):
            layer = nn.ModuleDict({
                "linear":  LinearLayer(self.dataSize+self.LLSize, self.LLSize)})#MLP([self.dataSize+self.LLSize, 7000, self.LLSize])}) #MLP([self.dataSize+self.LLSize, 500, self.LLSize])})
            self.layers.append(layer)

    def forward(self, Features:torch.tensor, labels:torch.tensor, Graph:torch.tensor, noisyOuts=False, **kwargs):         
        nAgents = Features.shape[0]
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = "cuda"

        # Random Initialization
        y = torch.distributions.Normal(0.0, 5).sample((nAgents, self.LLSize)).float().to(device)
        
        # Forward path
        outs = {}
        outs[0] = y
        indices = []
        for l in range(self.nLayers):
            if self.repeatLayers:
                layer = self.layers[l%self.coreLayers]
            else:
                layer = self.layers[l]
            idx = np.random.randint(0, Features.shape[1], self.batchSize)
            indices.append(idx)
            data = torch.cat((Features[:,idx], labels[:,idx]), dim=2).reshape((Features.shape[0],-1)).float().to(device)
            y1 = Graph @ y
            z = torch.cat((data, y), dim=1)
            y2 = layer["linear"](z)
            y2 = nn.ReLU()(y2)
            y = y1 - y2 
            # Noisy outputs
            if noisyOuts and l<self.nLayers-1:
                sigma = 1/(l+1)**2#torch.norm(compute_grad(x, kwargs['objective'], kwargs['test_dataset'], kwargs['device']), p=2, dim=1)
                y = y + torch.distributions.Normal(0.0, sigma).sample(y.shape).float().to(device)
            outs[l+1] = y
        torch.cuda.empty_cache()
        return y, outs, indices

    def forwardAsyn(self, Features, labels, Graph, nBOagents, BOflag=False, **kwargs):
        device = kwargs["device"]
        nAgents = Features.shape[0]
        if kwargs["BOagents"] is not None:
            BOagents = kwargs["BOagents"]
        else:
            BOflag = True
            BOagents = np.zeros((self.nLayers, nBOagents))

        # Random Initialization
        y = torch.distributions.Normal(0.0, 5).sample((nAgents, self.LLSize)).float().to(device)
        
        # Forward path
        outs = {}
        outs[0] = y
        indices = []
        for l in range(self.nLayers):
            if self.repeatLayers:
                layer = self.layers[l%self.coreLayers]
            else:
                layer = self.layers[l]
            if BOflag:
                BOagents[l] = np.random.choice(nAgents, nBOagents, replace=False)
            buffer = y[BOagents[l]]
            y[BOagents[l]] = 0.0
            idx = np.random.randint(0, Features.shape[1], self.batchSize)
            indices.append(idx)
            data = torch.cat((Features[:,idx], labels[:,idx]), dim=2).reshape((Features.shape[0],-1)).float().to(device)
            y1 = Graph @ y
            z = torch.cat((data, y), dim=1)
            y2 = layer["linear"](z)
            y2 = nn.ReLU()(y2)
            y = y1 - y2 
            y[BOagents[l]]= buffer
            outs[l+1] = y
        torch.cuda.empty_cache()
        return y, outs, indices, BOagents
