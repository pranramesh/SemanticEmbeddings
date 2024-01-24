import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.autograd import Variable

#### Alternative Idea ####
# 1. Given some text from user, make a call to cohere language model api and ask it to make the statement more negative in connotation (geenrate 5 samples)
# 2. re-embed this text using cohere embed model
# 3. visualize this more "negative" embedding in 3d space and compare it with original embedding (can try using PCA and t-SNE and see which gives better results)
# 4. can also train a sentiment classifier to classify original statement as positive or negative and see if it matches similarity to samples generated with similar connotation

class Visualizations():

    def __init__(self, data):
        self.data = data

    def principal_component_analysis(self, n):
        pca = PCA(n_components=n)
        m = np.array(self.data['Embedding'].tolist())
        pc = pca.fit_transform(m)
        return pc

    def stochastic_neighbors(self, n):
        tsne = TSNE(n_components=n, random_state=0)
        m = np.array(self.data['Embedding'].tolist())
        tsne_data = tsne.fit_transform(m)
        return tsne_data

#### Flow of Data in Application ####
# 1. user gives text input
# 2. Text input is then tokenized into vector
# 3. this tokenized text inpit gets passed through the encoder model 
# 4. The embedding is displayed
# 5. User can choose how to toggle the embedding.
# 6. The modified embedding is then passed through the decoder which gives back the modified text reconstructed purely from the latent space embedding
# 7. Interpolate and show outputs between more negative/positive statement and original statement

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class BetaVAE(nn.Module):
    #use cosin smilarity for loss metric
    def __init__(self, input_dim, hidden_dim, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim), 
            nn.ReLU(), 
            nn.Linear(in_features=hidden_dim, out_features=(2*latent_dim)), 
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim), 
            nn.ReLU(), 
            nn.Linear(in_features=hidden_dim, out_features=input_dim), 
        )

        self.beta = beta 

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = reparametrize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed