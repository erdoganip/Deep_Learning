##İpek Erdoğan
##2019700174
##Deep Learning HW1

import numpy as np
from Network import *
import time
import pickle
from eval import *
from tsne import *
def validation(network,arr,labels,batch_size):
    total_size = len(arr)
    indices = np.arange(total_size)
    epoch_loss = []
    correct = 0
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

if __name__ == '__main__':
    onehot_arr=[]
    dictionary_arr = np.load('data/vocab.npy')
    word_dictionary = {v: k for v, k in enumerate(dictionary_arr)}
    train_arr = np.load('data/train_inputs.npy')
    val_arr = np.load('data/valid_inputs.npy')
    test_arr = np.load('data/test_inputs.npy')
    train_labels = np.load('data/train_targets.npy')
    val_labels = np.load('data/valid_targets.npy')
    test_labels = np.load('data/test_targets.npy')

    for keys in word_dictionary:
        temp=np.zeros(len(word_dictionary))
        temp[keys]=1
        onehot_arr.append(temp)
    onehot_dictionary = {v: k for v, k in enumerate(onehot_arr)}

    network = Network(onehot_dictionary,learning_rate=0.001)
    batch_size=64
    total_size = len(train_arr)
    total_losses=[]
    epoch_num=1

    for epoch in range(epoch_num):
        print("EPOCH", epoch, "started")
        indices = np.random.permutation(total_size)
        epoch_loss=[]
        preds=[]
        correct=0
        for start_index in range(0, total_size, batch_size):
            end_index = total_size if start_index + batch_size > total_size else start_index + batch_size
            batch_input = [train_arr[i] for i in indices[start_index:end_index]]
            batch_label = [train_labels[i] for i in indices[start_index:end_index]]
            batch_loss , predictions = network.forward(batch_input,batch_label)
            network.backward()
            epoch_loss.append(batch_loss)
            preds = np.argmax(predictions, axis=1)
            correct += np.sum(preds==batch_label)
        loss = sum(epoch_loss)/len(epoch_loss)
        acc = 100*correct/total_size
        print("Epoch", epoch, "loss is: ", loss)
        print("Epoch", epoch, "accuracy is: ", acc)
        val_loss, val_acc = validation(network,val_arr,val_labels,batch_size)
        print("Epoch", epoch, "val loss is: ", val_loss)
        print("Epoch", epoch, "val accuracy is: ", val_acc)
        if epoch % 20 == 0:
            network.save_weights(epoch)

    test_filename="model.pk"
    test_loss, test_acc = eval(network,test_arr,test_labels,batch_size,test_filename)
    print("Test loss is: ",test_loss)
    print("Test accuracy is: ", test_acc)

    tsne_filename="model.pk"
    tsne_plot(tsne_filename, word_dictionary, onehot_dictionary)

    handmade_input_filename="model.pk"
    handmade_input=np.array([[133,17,84],[156,200,248],[169,191,248]])
    handmade_label=np.array([0,0,0])
    network.load_weights(handmade_input_filename)
    _, preds_ = network.forward(handmade_input, handmade_label)
    preds = np.argmax(preds_, axis=1)
    print("Results: ",preds)
    for i in preds:
        print(word_dictionary[i])
