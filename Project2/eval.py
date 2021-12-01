##Deep Learning HW2
##İpek Erdoğan
##2019700174
import numpy as np
from model import *
import time
import torch

def eval(net,testloader,PATH):
    batch_losses = []
    correct = 0
    #net = Net()
    #net.load_state_dict(torch.load(PATH))
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_size=0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            total_size += labels.size(0)
            outputs , flattens = net(inputs)
            flattens = flattens.detach().numpy()
            if (i==0):
                total_flatten = flattens
                total_labels = labels
            else:
                total_flatten = np.concatenate((total_flatten, flattens), axis=0)
                total_labels = np.concatenate((total_labels, labels), axis=0)
            loss = criterion(outputs, labels)
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            l_outputs = np.argmax(outputs.detach().numpy(), axis=1)
            correct += np.sum(l_outputs == labels.detach().numpy())
        acc = 100 * correct / total_size
        total_loss = sum(batch_losses) / len(batch_losses)
    return total_flatten, total_labels, total_loss,acc