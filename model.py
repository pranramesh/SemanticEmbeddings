# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#### Alternative Idea ####
# 1. Given some text from user, make a call to cohere language model api and ask it to make the statement more negative in connotation
# 2. re-embed this text using cohere embed model
# 3. visualize this more "negative" embedding in 3d space and compare it with original embedding (can try using PCA and t-SNE and see which gives better results)

class Visualizations():

    def __init__(self, original_embedding, modified_embedding):
        self.original_embedding = original_embedding
        self.modified_embedding = modified_embedding

    def principal_component_analysis(self):
        pca = PCA(n_components=2)
        og = pca.fit_transform(np.array([self.original_embedding]))
        mod = pca.fit_transform(np.array([self.new_embedding]))
        return [og, mod]

    # def stochastic_neighbords(self, )




#### Flow of Data in Application ####
# 1. user gives text input
# 2. Text input is then tokenized into vector
# 3. this tokenized text inpit gets passed through the encoder model 
# 4. The embedding is displayed
# 5. User can choose how to toggle the embedding.
# 6. The modified embedding is then passed through the decoder which gives back the modified text reconstructed purely from the latent space embedding

# class Autoencoder(nn.Module):
#     #use cosin smilarity for loss metric
#     def __init__(self, input_dim, embedding_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(in_features=input_dim, out_features=embedding_dim), 
#             nn.ReLU(), 
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim), 
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim), 
#             nn.ReLU(), 
#             nn.Linear(in_features=embedding_dim, out_features=input_dim), 
#         )
#     def forward(self, x):
#         embedding = self.encoder(x)
#         reconstructed = self.decoder(embedding)
#         return reconstructed