from mmvec.scheduler import AlternatingStepLR
import torch
import torch.nn as nn
import torch.optim as optim


import unittest

class TestScheduler(unittest.TestCase):

    def setUp(self):
        layers = []
        layers.append(nn.Linear(3, 4))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(4, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)


    def test_scheduler(self):
        optimizer = optim.Adam([
            {'params': self.net[0].parameters(), 'lr': 0.1},
            {'params': self.net[1].parameters(), 'lr': 0.1}
        ])

        scheduler = AlternatingStepLR(optimizer, 10)
        left_lr = []
        right_lr = []
        for _ in range(100):
            x = torch.randn(3)
            y = torch.round(torch.rand(1))
            self.net.train()
            optimizer.zero_grad()
            yhat = self.net.forward(x)
            loss = (y - yhat) ** 2
            loss.backward()
            optimizer.step()
            scheduler.step()
            left_lr.append(optimizer.param_groups[0]['lr'])
            right_lr.append(optimizer.param_groups[1]['lr'])

        # test to make sure first step alternates
        self.assertAlmostEqual(left_lr[0], 1e-8)
        self.assertAlmostEqual(right_lr[0], 0.1)
        self.assertAlmostEqual(left_lr[1], 0.1)
        self.assertAlmostEqual(right_lr[1], 1e-8)

        # test to make sure second step alternates
        self.assertAlmostEqual(left_lr[10], 1e-8)
        self.assertAlmostEqual(right_lr[10], 0.01)
        self.assertAlmostEqual(left_lr[11], 0.01)
        self.assertAlmostEqual(right_lr[11], 1e-8)


if __name__ == "__main__":
    unittest.main()
