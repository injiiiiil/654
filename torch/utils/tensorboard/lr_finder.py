from copy import deepcopy
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch

from .writer import SummaryWriter

class Smoothener:
    "Create a smooth moving average for a value (loss, etc) using `beta`."
    def __init__(self, beta=0.98):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def smoothen(self, val):
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        return self.mov_avg / (1 - self.beta ** self.n)

class LRFinder:
    def __init__(self,
                 model,
                 train_dl,
                 opt,
                 loss_func,
                 start_lr=1e-7,
                 end_lr=10.,
                 num_it=100,
                 log_dir=None):
        self.model, self.train_dl, self.loss_func, self.num_it = model, train_dl, loss_func, num_it
        self.has_run, self.losses, self.lrs = False, [], []
        self.log_dir, self.num_runs = log_dir, 0
        self.opt, self.scheduler = self.__get_opt_and_scheduler(opt, start_lr, end_lr)

    def __annealing_exp(self, pct, start, end):
        "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end/start) ** pct

    def __get_opt_and_scheduler(self, opt, start_lr, end_lr):
        new_opt = deepcopy(opt)
        for g in new_opt.param_groups:
            g['initial_lr'] = start_lr
            g['lr'] = start_lr
        anneal_func = partial(self.__annealing_exp, start=start_lr, end=end_lr)
        schedule_func = lambda step: anneal_func(step/float(self.num_it))
        scheduler = LambdaLR(new_opt, schedule_func)
        return new_opt, scheduler

    def __split_list(self, vals, skip_start, skip_end):
        return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]

    def find(self):
        self.has_run = False
        # Save model initial state
        torch.save(self.model.state_dict(), "tmp")
        self.model.train()

        # Train model and record loss vs. lr
        smoothener = Smoothener()
        self.losses = []
        self.lrs = []
        for i, (xb, yb) in enumerate(self.train_dl):
            if i >= self.num_it:
                break

            for g in self.opt.param_groups:
                lr = g.get('lr')
                if lr is not None:
                    break
            loss = self.loss_func(self.model, xb, yb)
            smooth_loss = smoothener.smoothen(loss.item())
            self.losses.append(smooth_loss)
            self.lrs.append(lr)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.scheduler.step()

        # Reload model initial state
        self.model.load_state_dict(torch.load("tmp"))
        self.model.eval()
        self.has_run = True

    def plot(self, skip_start=10, skip_end=5, push_to_tensorboard=True):
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
        if not self.has_run:
            raise Exception("You need to run find() first!")
        lrs = self.__split_list(self.lrs, skip_start, skip_end)
        losses = self.__split_list(self.losses, skip_start, skip_end)
        fig, ax = plt.figure(), plt.gca()
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if push_to_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)
            writer.add_figure("lr_finder/run-{}".format(self.num_runs), fig, self.num_runs)
            writer.close()
            self.num_runs += 1
