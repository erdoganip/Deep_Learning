# This is a sample Python script.
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     ])

if __name__ == '__main__':
    batch_size = 256
    epoch_no = 51
    trainset = torchvision.datasets.MNIST(root='mnist_data/', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    frame_dim = 28
    lstm_hidden_dim = 128
    feature_dim = 128
    encoder = Encoder(frame_dim, lstm_hidden_dim, feature_dim)
    encoder.to(device)
    decoder = Decoder(feature_dim)
    decoder.to(device)
    reconstruction_criterion = nn.BCELoss()
    optimizerE = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizerD = optim.Adam(decoder.parameters(), lr=0.001, betas=(0.9, 0.999))

    total_train_recon = []
    total_train_KL = []
    total_train_loss = []
    for epoch in range(epoch_no):  # loop over the dataset multiple times
        batch_losses = []
        batch_recon = []
        batch_KL = []
        encoder.train()
        decoder.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = torch.squeeze(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizerE.zero_grad()
            optimizerD.zero_grad()
            mean, log_var = encoder(inputs)
            feature = reparameterize(mean, log_var)
            sampled = decoder(feature)
            sampled = sampled.squeeze()
            loss = reconstruction_criterion(sampled, inputs) + KL_divergence(mean, log_var)
            loss.backward()
            optimizerE.step()
            optimizerD.step()
            batch_loss = loss.item()
            recons_value = reconstruction_criterion(sampled, inputs).item()
            regularization_value = KL_divergence(mean, log_var).item()
            batch_losses.append(batch_loss)
            batch_recon.append(recons_value)
            batch_KL.append(regularization_value)

        total_train_loss.append(sum(batch_losses) / len(batch_losses))
        total_train_KL.append(sum(batch_KL) / len(batch_KL))
        total_train_recon.append(sum(batch_recon) / len(batch_recon))
        print("Epoch ", epoch)
        print("Epoch Total Loss: ", sum(batch_losses) / len(batch_losses))
        print("Epoch Reconstruction Loss: ", sum(batch_recon) / len(batch_recon))
        print("KL Term: ", sum(batch_KL) / len(batch_KL))

        if epoch % 10 == 0:
            model_save_name = 'classifier.pt'
            PATH = os.path.join('Encoder_' + str(epoch) + '.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizerE.state_dict(),
            }, PATH)
            PATH = os.path.join('Decoder_' + str(epoch) + '.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
            }, PATH)

    epochs = range(0, epoch_no)
    plt.plot(epochs, total_train_loss, 'g', label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("total_loss" + str(epoch) + ".png")
    plt.show()

    plt.plot(epochs, total_train_recon, 'g', label='BCE Loss')
    plt.title('BCE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("bce_loss" + str(epoch) + ".png")
    plt.show()

    plt.plot(epochs, total_train_KL, 'g', label='KL Divergence')
    plt.title('Regularization Term')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("kl_div" + str(epoch) + ".png")
    plt.show()