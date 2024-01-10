import torch
import torch.nn as nn
import torch.optim as optim


#### Flow of Data in Application ####
# 1. user gives text input
# 2. Text input is then tokenized into vector
# 3. this tokenized text inpit gets passed through the encoder model 
# 4. The embedding is displayed
# 5. User can choose how to toggle the embedding.
# 6. The modified embedding is then passed through the decoder which gives back the modified text reconstructed purely from the latent space embedding

class Autoencoder(nn.Module):
    #use cosin smilarity for loss metric
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=embedding_dim)
            nn.ReLU()
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
            nn.ReLU()
            nn.Linear(in_features=embedding_dim, out_features=input_dim)
            nn.ReLU()
        )
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed
