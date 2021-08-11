import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import tqdm, random
# from sklearn.ensemble import RandomForestClassifier
# from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.optimize import AdamOptimizer

from base.testproblem.testfunction import Rastrigin
from base.kde.toy_kde import NaiveKDE



def kdeBackend(cls):
    def wrapper(f):
        def deco(*args, **kwargs):
            f(*args, cls=cls, **kwargs)

        return deco

    return wrapper


class FloatVarKDE():
    def __init__(self, low=0, high=1, bandwidth=0.1):
        self.bandwidth = bandwidth
        # self.kde = NaiveKDE(bandwidth=bandwidth)
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.low = low
        self.high = high
        # self.kde.fit(high-low)

    def sample(self, num):
        cnt = 0
        samples = []
        while cnt < num:
            sample = self.kde.sample().item()
            if self.low <= sample <= self.high:
                cnt += 1
                samples.append(sample)
        return np.asarray(samples)

    def fit(self, points):
        self.kde.fit(points.reshape(-1, 1))

    def log_pdf(self, x):
        return self.kde.score_samples(x.reshape(-1, 1))


class FloatNaiveKDE(NaiveKDE):
    def __init__(self, low=0, high=1, bandwidth=0.1):
        NaiveKDE.__init__(self, bandwidth=bandwidth)
        self.low = low
        self.high = high
        self.bandwidth = bandwidth

    def fit(self, x):
        NaiveKDE.fit(self, x)

    def log_pdf(self, x):
        return np.log(NaiveKDE.evaluate(self, x))

    def sample(self, num):
        ct = 0
        samples = []
        while ct < num:
            u = np.random.uniform(0, 1)
            idx = int(u * len(self.data))
            sample = np.random.normal(self.data[0, idx].item(), self.bandwidth)
            if self.low <= sample <= self.high:
                ct += 1
                samples.append(sample)

        return samples
        # u = np.random.random.uniform(0, 1, size=num)
        # idx = (u * len(self.data)).as_type(np.int64)
        # return np.random.normal(self.data[idx], self.bandwidth)


class ToyTPE():
    # @kdeBackend(FloatVarKDE)
    def __init__(self,
                 config,
                 cls=FloatVarKDE,
                 bandwidth=1,
                 min_sample=30,
                 gamma=0.1,
                 candidates_num=24):
        self.gamma = gamma
        self.min_sample = min_sample
        self.bandwidth = bandwidth
        self.candidates_num = candidates_num
        self.hp_num = len(config)
        self.hp_range = []
        self.hp_names = list(sorted(config.keys()))
        for name in self.hp_names:
            self.hp_range.append(config[name])
        self.lx = [cls(*lh, bandwidth=bandwidth) for lh in (self.hp_range)]
        self.gx = [cls(*lh, bandwidth=bandwidth) for lh in (self.hp_range)]
        # self.lx = [FloatNaiveKDE(*lh) for lh in (self.hp_range)]
        # self.gx = [FloatNaiveKDE(*lh) for lh in (self.hp_range)]
        self.history = []
        self.history_y = []

    def suggest(self, ):
        # 只suggest 一个
        if len(self.history_y) <= self.min_sample or np.random.rand() < 0.1:
            return self._random_suggest()
        else:
            suggest_array = []
            for i in range(self.hp_num):
                candidates = self.lx[i].sample(self.candidates_num)
                lx = self.lx[i].log_pdf(candidates)
                gx = self.gx[i].log_pdf(candidates)
                max_idx = np.argmax(lx - gx)
                suggest_array.append(candidates[max_idx])
            return {k: suggest_array[i] for i, k in enumerate(self.hp_names)}

    def _random_suggest(self):
        suggest_array = []
        for i in range(self.hp_num):
            suggest_array.append((np.random.rand() *
                                  (self.hp_range[i][1] - self.hp_range[i][0]) +
                                  self.hp_range[i][0]))
        return {k: suggest_array[i] for i, k in enumerate(self.hp_names)}

    def _split_l_g(self):
        samples = np.asarray(self.history)
        good_num = int(self.gamma * len(samples))
        samples_sorted = samples[np.argsort(self.history_y)]
        good_points = samples_sorted[:good_num, :]
        bad_points = samples_sorted[good_num:, :]
        return good_points, bad_points

    def observe(self, x, y):
        sample_array = []
        for k in self.hp_names:
            sample_array.append(x[k])
        self.history.append(sample_array)
        self.history_y.append(y)
        if len(self.history_y) < self.min_sample:
            return

        good_points, bad_points = self._split_l_g()
        for i in range(self.hp_num):
            self.lx[i].fit(good_points[:, i])
            self.gx[i].fit(bad_points[:, i])

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        self.dense = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.dense(x)
    
    def fit(self, x, y):
        self.forward(x)
        for iter in range(20):
            self.forward(x)

class ToyBORE():
    def __init__(self, config, gamma=0.5, min_sample=30, candidates_num=24):
        self.gamma = gamma
        self.min_sample = min_sample
        self.history = []
        self.history_y = []
        self.hp_num = len(config)
        self.candidates_num = candidates_num
        self.hp_range = []
        self.hp_names = list(sorted(config.keys()))
        for name in self.hp_names:
            self.hp_range.append(tuple(config[name]))
        self.bounds = np.asarray(self.hp_range)
        self.classify = MLP(self.hp_num)
        # self.labels = []
    def argmax(self):
        f_min = np.inf
        # result = None
        for i in range(self.candidates_num):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
        return result

    def _random_suggest(self):
        suggest_array = []
        for i in range(self.hp_num):
            suggest_array.append((np.random.rand() *
                                  (self.hp_range[i][1] - self.hp_range[i][0]) +
                                  self.hp_range[i][0]))
        return {k: suggest_array[i] for i, k in enumerate(self.hp_names)}

    def suggest(self):
        if len(self.history_y) <= self.min_sample or np.random.rand() < 0.1:
            return self._random_suggest()
        else:
            suggest_array = self.argmax()
            return {k: suggest_array[i] for i, k in enumerate(self.hp_names)}

    def observe(self, x, y):
        sample_array = []
        for k in self.hp_names:
            sample_array.append(x[k])
        self.history.append(sample_array)
        self.history_y.append(y)
        if len(self.history_y) < self.min_sample:
            return
        labels = np.zeros(len(self.history_y))
        labels[np.argsort(self.history_y)[:int(self.gamma *
                                               len(self.history_y))]] = 1
        self.classify.fit(torch.tensor(self.history),
                          torch.tensor(labels))


def main():
    np.random.seed(42)
    random.seed(42)
    func = Rastrigin(dim=2)
    opt = ToyTPE(config=func._load_api_config(), cls=FloatVarKDE)
    # opt = ToyBORE(config=func._load_api_config())
    for iter in tqdm.tqdm(range(100)):
        x = opt.suggest()
        y = func(x)
        opt.observe(x, y)
    plt.figure()
    plt.plot(opt.history_y)
    plt.show()

    print('0')


if __name__ == '__main__':
    main()