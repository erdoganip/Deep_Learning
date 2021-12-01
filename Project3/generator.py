##Deep Learning HW3
##İpek Erdoğan
##2019700174
import numpy as np
from model import *
import torch
import random
random.seed(42)
torch.random.seed()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_dim=128
PATH='Decoder_50.pt'
decoder_x=Decoder(feature_dim)
checkpoint= torch.load(PATH, map_location=torch.device('cpu'))
decoder_x.load_state_dict(checkpoint['model_state_dict'])
decoder_x.eval()
decoder_x.to(device)

for i in range(10):
  torch.random.seed()
  eval_input=torch.randn(128).to(device)
  #eval_input = torch.tensor(np.random.rand(128))
  #eval_input = eval_input.to(device)
  sample = decoder_x(eval_input.float()).squeeze()
  sample = sample.cpu().detach().numpy() # convert images to numpy for display
  plt.imshow(sample,cmap='gray')
  plt.show()