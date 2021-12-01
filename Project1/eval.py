##İpek Erdoğan
##2019700174
##Deep Learning HW1

import numpy as np
from Network import *
import time
def eval(network,arr,labels,batch_size,filename):
    total_size = len(arr)
    indices = np.arange(total_size)
    epoch_loss = []
    correct = 0
    network.load_weights(filename)
    for start_index in range(0, total_size, batch_size):
        end_index = total_size if start_index + batch_size > total_size else start_index + batch_size
        batch_input = [arr[i] for i in indices[start_index:end_index]]
        batch_label = [labels[i] for i in indices[start_index:end_index]]
        batch_loss, predictions = network.forward(batch_input, batch_label)
        epoch_loss.append(batch_loss)
        preds = np.argmax(predictions, axis=1)
        correct += np.sum(preds == batch_label)
    loss = sum(epoch_loss) / len(epoch_loss)
    acc = 100*correct / total_size
    return loss,acc