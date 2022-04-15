import torch
import torch.nn as nn
from torch.distributions import Multinomial

class MMvec(nn.Module):
    def __init__(self, num_microbes, num_metabolites, latent_dim):
        super().__init__()

        self.encoder = nn.Embedding(num_microbes, latent_dim)
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, num_metabolites),
                nn.Softmax(dim=2)
                )

    def forward(self, X, Y):
        z = self.encoder(X)
        y_pred = self.decoder(z)
        
        forward_dist = Multinomial(total_count=0,
                                   validate_args=False,
                                   probs=y_pred)

        forward_dist = forward_dist.log_prob(Y)

        lp = forward_dist.mean(0).mean()

        return lp
