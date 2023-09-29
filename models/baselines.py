import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import logging
import os

from models.ResNet import *

def centralized_training(metadataset, criterion, args):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.Dataset == 'CIFAR10':
        checkpoint = torch.load('./checkpoint/ckpt18VHB.pth')
        inDim = 3
        outDim = 10
    elif args.Dataset == 'CIFAR100':
        checkpoint = torch.load('./checkpoint/ckptCIFAR100.pth')
        inDim = 3
        outDim = 100
    elif args.Dataset == 'MNIST':
        checkpoint = torch.load('./checkpoint/ckptMNIST.pth')
        inDim = 1
        outDim = 10

    testAccuracy2 = []
    testAccuracy15 = []
    nTrain = int(args.nTrainPerAgent * args.nAgents)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for ibatch in range(len(metadataset)):
        # Load the CNN model
        CNN2 = ResNetConv18(inDim)
        CNN2.to(device)
        if device == 'cuda':
            CNN2 = torch.nn.DataParallel(CNN2)
            cudnn.benchmark = True
            optimizer = optim.SGD(CNN2.module.linear.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(CNN2.linear.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        CNN2.load_state_dict(checkpoint['net'])
        
        # Train the CNN model
        CNN2.train()
        best = np.inf
        for epoch in range(3000):
            images = torch.reshape(metadataset[ibatch][0][0], (-1, 512)).float().to(device)
            targets = torch.reshape(metadataset[ibatch][0][1], (-1, args.nClasses)).long()
            optimizer.zero_grad()                
            logits = CNN2.module.forwardLast(images[:nTrain])
            loss = criterion(logits.cpu(), torch.argmax(targets, dim=1))
            outputs = torch.max(logits.cpu(), axis=1)[1]
            acc = torch.sum(outputs == torch.argmax(targets, dim=1))/targets.shape[0]
            loss.backward()
            optimizer.step()
            print('Accuracy at epoch ', epoch, ': ', acc.item())
            
            if epoch == 10:
                torch.save({'net':CNN2.state_dict(), 'epoch':epoch}, f'./checkpoint/CNN2_{ibatch}_15.pth')

        torch.save({'net':CNN2.state_dict(), 'epoch':epoch}, f'./checkpoint/CNN2_{ibatch}.pth')
        # Evaluate the CNN model
        CNN2.eval()
        images = torch.reshape(metadataset[ibatch][1][0], (-1, 512)).float().to(device)
        targets = torch.reshape(metadataset[ibatch][1][1], (-1,args.nClasses)).long()
        logits = CNN2.module.forwardLast(images).cpu()
        outputs = torch.max(logits, axis=1)[1]
        testAccuracy2.append(torch.sum(outputs == torch.argmax(targets, dim=1))/targets.shape[0])

        # At 15:
        checkpoint = torch.load(f'./checkpoint/CNN2_{ibatch}_15.pth')
        CNN2.load_state_dict(checkpoint['net'])
        CNN2.eval()
        images = torch.reshape(metadataset[ibatch][1][0], (-1, 512)).float().to(device)
        targets = torch.reshape(metadataset[ibatch][1][1], (-1,args.nClasses)).long()
        logits = CNN2.module.forwardLast(images).cpu()
        outputs = torch.max(logits, axis=1)[1]
        testAccuracy15.append(torch.sum(outputs == torch.argmax(targets, dim=1))/targets.shape[0])
    logging.debug(r'Accuracy at the end {:.2f} +/- {:.2f}'.format(np.mean(testAccuracy2)*100, np.std(testAccuracy2)*100))
    logging.debug(r'Accuracy (truncated) {:.2f} +/- {:.2f}'.format(np.mean(testAccuracy15)*100, np.std(testAccuracy15)*100))

    return np.mean(testAccuracy2)*100, np.std(testAccuracy2)*100
