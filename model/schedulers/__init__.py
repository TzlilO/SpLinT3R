

import math
from torch.optim.lr_scheduler import _LRScheduler


class DecayingCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, alpha=0.5, beta=0.5,
                 verbose=False, start_decay_cycle=0, stop_decay_cycle=float('inf'), valid_groups=['f_rest', 'scale_params']):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0: First cycle step size.
            T_mult: Cycle steps magnification. Default: 1.
            eta_min: Minimum learning rate. Default: 0.
            last_epoch: The index of last epoch. Default: -1.
            alpha: Decay factor for max_lr after each restart. Default: 0.5.
            beta: Decay factor for min_lr after each restart. Default: 0.5.
            verbose: If True, prints a message to stdout for each update. Default: False.
            start_decay_cycle: Cycle number to start decay. Default: 0 (start immediately).
            stop_decay_cycle: Cycle number to stop decay. Default: inf (never stop).
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not 0 <= alpha < 1:
            raise ValueError(f"Expected 0 <= alpha < 1, but got {alpha}")
        if not 0 <= beta < 1:
            raise ValueError(f"Expected 0 <= beta < 1, but got {beta}")
        if not isinstance(start_decay_cycle, int) or start_decay_cycle < 0:
            raise ValueError(f"Expected non-negative integer start_decay_cycle, but got {start_decay_cycle}")
        if not (isinstance(stop_decay_cycle, (int, float)) and stop_decay_cycle > start_decay_cycle):
            raise ValueError(f"Expected stop_decay_cycle > start_decay_cycle, but got {stop_decay_cycle}")

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.base_eta_min = eta_min
        self.alpha = alpha
        self.beta = beta
        self.cycle = 0
        self.T_cur = last_epoch
        self.verbose = verbose
        self.start_decay_cycle = start_decay_cycle
        self.stop_decay_cycle = stop_decay_cycle
        self.valid_param_groups = valid_groups
        self.base_max_lrs = [group['lr'] for group in optimizer.param_groups]
        self.eta_min = self.base_eta_min
        self.base_lrs = [lr for lr in self.base_max_lrs]
        super(DecayingCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur == 0:
            return self.calculate_lr()
        elif (self.T_cur + 1) % self.T_i == 0:
            return self.calculate_lr(is_restart=True)
        else:
            return self.calculate_lr()

    def calculate_lr(self, is_restart=False):
        cycle_factor = 1
        if self.start_decay_cycle <= self.cycle < self.stop_decay_cycle:
            cycle_factor = self.alpha ** (self.cycle - self.start_decay_cycle + int(is_restart))

        return [
            self.eta_min + (max_lr * cycle_factor - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for max_lr in self.base_max_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        if self.start_decay_cycle <= self.cycle < self.stop_decay_cycle:
            self.eta_min = self.base_eta_min * (self.beta ** (self.cycle - self.start_decay_cycle))
        else:
            self.eta_min = self.base_eta_min

        self.last_epoch = math.floor(epoch)

        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            if param_group['name'] not in self.valid_param_groups:
                continue
            param_group['lr'] = lr

        if self.verbose:
            print(f'Epoch {self.last_epoch}: adjusting learning rate to {new_lrs}')

