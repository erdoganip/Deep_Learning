##Deep Learning HW2
##İpek Erdoğan
##2019700174

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model import *
import torch.optim as optim
from eval import *
import os
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomCrop(size=[32,32], padding=4),
     #transforms.RandomVerticalFlip(p=0.5),
    ])

transform2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

if __name__ == '__main__':
    batch_size=16
    epoch_no=50
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainset = torchvision.datasets.CIFAR10(root='cifar10_data/', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='cifar10_data/', train=False,
                                           download=True,transform=transform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    net = Net()

    PATH = './cifar_net.pth'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    #optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99)

    total_train_acc=[]
    total_train_loss=[]
    total_test_acc = []
    total_test_loss = []
    for epoch in range(epoch_no):
        batch_losses=[]
        correct=0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs,_ = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            l_outputs = np.argmax(outputs.detach().numpy(), axis=1)
            correct += np.sum(l_outputs == labels.detach().numpy())
        epoch_acc = 100 * correct / len(trainset)
        total_train_acc.append(epoch_acc)
        epoch_loss = sum(batch_losses) / len(batch_losses)
        total_train_loss.append(epoch_loss)
        flattens, test_labels, epoch_test_loss,epoch_test_acc = eval(net,testloader, PATH)
        total_test_acc.append(epoch_test_acc)
        total_test_loss.append(epoch_test_loss)

        if ( epoch % 20 == 0 or epoch == 49):
            PATH = os.path.join('net_' + str(epoch) + '.pt')
            torch.save(net.state_dict(), PATH)

            tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000, random_state=23)
            new_values = tsne_model.fit_transform(flattens)

            xxx = sns.scatterplot(new_values[:, 0], new_values[:, 1], hue=test_labels, legend='full', palette=palette)
            plt_path = os.path.join('latent_' + str(epoch) + '.png')
            plt.savefig(plt_path)
            plt.show()
            #latent_path = os.path.join('latent_' + str(epoch) + '.npy')
            #with open(latent_path, 'wb') as f:
            #    np.save(f, np.array(flattens))

        print("Epoch ",epoch)
        print("Epoch Accuracy: ",epoch_acc)
        print("Epoch Loss: ", epoch_loss)
        print("Epoch Test Accuracy: ", epoch_test_acc)
        print("Epoch Test Loss: ", epoch_test_loss)

    epochs = range(0,epoch_no)

    plt.plot(epochs, total_train_loss, 'g', label='Training Loss')
    plt.plot(epochs, total_test_loss, 'b', label='Testing Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    plt.plot(epochs, total_train_acc, 'g', label='Training Accuracy')
    plt.plot(epochs, total_test_acc, 'b', label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("acc.png")
    plt.show()

