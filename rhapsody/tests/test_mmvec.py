import unittest


class TestMMvec(unittest.TestCase):

    def setUp(self):

        self.num_microbes = 100
        self.num_metabolites = 20
        self.num_samples=100
        self.latent_dim=2
        self.means = (-3, 3)
        self.microbe_total=300
        self.metabolite_total=900
        self.uB=0
        self.sigmaB=1
        self.sigmaQ=2
        self.uU=0
        self.sigmaU=0.01
        self.uV=0
        self.sigmaV=0.01
        self.s = 0.2
        self.seed=1        
        self.eUun = 0.2
        self.eVun = 0.2


        res = random_bimodal(num_microbes=self.num_microbes, 
                             num_metabolites=self.num_metabolites, 
                             num_samples=self.num_samples,
                             latent_dim=self.latent_dim, 
                             means=self.means,
                             microbe_total=self.microbe_total, 
                             metabolite_total=self.metabolite_total,
                             sigmaQ=0.1,
                             uU=0, 
                             sigmaU=1, 
                             uV=0, 
                             sigmaV=1,
                             seed=0):        
        (self.microbe_counts, 
         self.metabolite_counts, 
         self.eUmain, 
         self.eVmain, 
         self.eUbias, 
         self.eVbias) = self

    def test_mmvec_loss(self):
        pass


if __name__ == "__main__":
    unittest.main()
