from rhapsody.layers import GaussianEmbedding, GaussianDecoder
from tqdm import tqdm
from abc import ABC, abstractmethod
from batch import get_batch


class VEC(nn.Module, ABC):

    def __init__(self, batch_size, subsample_size):
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        super(VEC, self).__init__()

    @abstractmethod    
    def loss(self):
        print('Loss function not implemented.')

    def fit(self, trainX, trainY, testX, testY):
        best_loss = np.inf
        losses = []
        klds = []
        likes = []
        errs = []
        for ep in tqdm(range(0, epochs)):
        
            model.train()
            scheduler.step()
            for i in range(0, num_samples, self.batch_size):
                optimizer.zero_grad()
        
                inp, out = get_batch(trainX, trainY, i % num_samples,
                                     self.subsample_size, self.batch_size)
        
                pred = self.forward(inp)            
                loss, kld, like = self.loss(pred, out)      
        
                err = torch.mean(torch.abs(F.softmax(pred, dim=1) * metabolite_total - out))
                loss.backward()
                                 
                errs.append(err.item())
                losses.append(loss.item())
                klds.append(kld.item())
                likes.append(like.item())

                optimizer.step()


class MMvec(VEC):
    def __init__(self, num_samples, num_microbes, num_metabolites, microbe_total,
                 latent_dim, batch_size=10, subsample_size=100, mc_samples=10,
                 device='cpu'):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.mc_samples = mc_samples
        self.microbe_total = microbe_total
        # TODO: enable max norm in embedding to account for scale identifiability
        self.encoder = GaussianEmbedding(num_microbes, latent_dim)
        self.decoder = GaussianDecoder(latent_dim, num_metabolites)
        
    def forward(self, x):
        code = self.encoder(x)
        log_probs = self.decoder(code)
        #zeros = torch.zeros(self.batch_size * self.subsample_size, 1)
        #log_probs = torch.cat((zeros, alrs), dim=1)
        return log_probs

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        mean_like = torch.zeros(mc_samples, device=self.device)  
        kld = self.encoder.divergence() + self.decoder.divergence()
        n = self.microbe_total * self.num_samples
        likelihood = n * torch.mean(Multinomial(logits=pred).log_prob(obs))
        elbo = kld + likelihood
        return -elbo, kld, likelihood      

    
