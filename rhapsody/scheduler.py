from torch.optim.lr_scheduler import _LRScheduler
import warnings


class AlternatingStepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, baseline_lr=1e-8, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.baseline_lr = 1e-8
        self.alternate = 0
        super(AlternatingStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        n_layers = len(self.optimizer.param_groups)

        # only step for 1 layer at a time
        for i, (param_group, lr) in enumerate(zip(
                self.optimizer.param_groups, self.get_lr())):
            if i % n_layers == self.alternate:
                param_group['lr'] = lr
            else:
                param_group['lr'] = self.baseline_lr
        self.alternate = (self.alternate + 1) % n_layers
