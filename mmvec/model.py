import torch
import torch.nn as nn
from torch.distributions import Multinomial, Normal

from torch.nn.parallel import DistributedDataParallel as ddp

class MMvec(nn.Module):
    def __init__(self, num_microbes, num_metabolites, latent_dim, sigma_u,
            sigma_v):
        super().__init__()

        self.encoder = nn.Embedding(num_microbes, latent_dim)
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, num_metabolites),
                nn.Softmax(dim=2)
                )
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

    def forward(self, X, Y):
        # Three likelihoods, the likelihood of each weight and the likelihood
        # of the data fitting in the way that we thought
        # LY
        z = self.encoder(X)
        y_pred = self.decoder(z)

        forward_dist = Multinomial(total_count=0,
                                   validate_args=False,
                                   probs=y_pred)

        forward_dist = forward_dist.log_prob(Y)

        l_y = forward_dist.mean(0).mean()

        # LU
        u_weights = self.encoder.weight#.detach().numpy()
        l_u = Normal(0, self.sigma_u).log_prob(u_weights).sum()
        #l_u = torch.normal(0, self.sigma_u).log_prob(z

        # LV
        # index zero currently holds "linear", may need to be changed later
        v_weights = self.decoder[0].weight#.detach().numpy()
        l_v = Normal(0, self.sigma_v).log_prob(v_weights).sum()

        likelihood_sum = l_y + l_u + l_v

        return likelihood_sum


def mmvec_training_loop(microbes, metabolites, model, optimizer, batch_size, epochs):

    for epoch in range(epochs):

        draws = torch.multinomial(microbes,
                                  batch_size,
                                  replacement=True).T

        mmvec_model = model(draws, metabolites)

        optimizer.zero_grad()
        mmvec_model.backward()
        optimizer.step()

#        if epoch % 5 == 0:
#            print(f"loss: {mmvec_model.item()}\nBatch #: {epoch}")
