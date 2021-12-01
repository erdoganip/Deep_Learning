##Deep Learning HW3
##İpek Erdoğan
##2019700174
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_s, hidden_size, output_size, drop_prob=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_s
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear_mean = nn.Linear(self.hidden_size,self.output_size)
        self.linear_std = nn.Linear(self.hidden_size,self.output_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, dropout=drop_prob)

    def forward(self, input):
        lstm_out, hidden = self.lstm(input)
        out = lstm_out[:,-1,:]
        mean_out = self.linear_mean(out)
        log_var_out = self.linear_std(out)

        return mean_out,log_var_out

class Decoder(nn.Module):
    def __init__(self, em_dim):
        super(Decoder, self).__init__()
        self.nf = 8
        self.im_dim = 7
        self.em_dim = em_dim

        self.trans = nn.Sequential(
            nn.Linear(self.em_dim, self.nf * 4 * self.im_dim * self.im_dim),
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 1,),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(16, 8, 4, 2,),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(8, 4, 4, 1,),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(4, 1, 4, 1,),
            nn.Sigmoid()
        )
    def forward(self, f_v):
        hidden = f_v.view(-1,self.em_dim)
        small = self.trans(hidden).view(-1, self.nf * 4, self.im_dim, self.im_dim)
        img = self.main(small)
        return img

def KL_divergence(mean,log_var):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    return -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
    return mean + eps * std

