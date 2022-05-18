#! python3

import numpy as np
import os
import logging
from sympy import Symbol
from sympy.solvers import solve
import sympy
from math import factorial
from scipy.stats import chi2
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from statistics import mean, variance

# logging.disable(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Poisson:
    def __init__(self, x, y, beta):
        self.x = x
        self.xc = x.shape[1]
        self.y = y
        self.beta = beta
        self.N = len(y)
        self.p = len(beta)

    def mu(self, beta):
        self.__mu = np.array([sympy.exp(xb[0]) for xb in self.x.dot(beta)])
        return np.array([sympy.exp(xb[0]) for xb in self.x.dot(beta)])
        
    def log_likelihood(self, beta):
        y_factorials = [float(factorial(int(yy))) for yy in y]
        x_dot_beta = x.dot(beta)
        muu = self.mu(beta)
        return sum(-muu[i] + y[i]*x_dot_beta[i] - np.log(y_factorials[i]) for i, _ in enumerate(y))[0]

    def score_statistic(self, beta):
        muu = self.mu(beta)
        return [sum(self.x[i, j]*(self.y[i] - muu[i]) for i in range(self.N)) for j in range(self.xc)]

    def solve_for_beta(self, u, initial_guess):
        soln = sympy.nsolve(u, self.beta.flatten(), initial_guess)
        betas = {b: bb for b, bb in zip(self.beta.flatten(), soln)}
        return betas

    def goodness_of_fit(self, beta_estimate):
        df = self.N - self.p
        chi2crit = chi2.ppf(0.95, df)
        muu = self.mu(self.beta)
        mu_hat = np.array([muu[i].subs(beta_estimate) for i, _ in enumerate(y)])
        deviance_statistic = 2 * sum((- (self.y[i]-mu_hat[i])) + (self.y[i] * np.log(float(self.y[i]/mu_hat[i]))
                                        if self.y[i] > 0 else 0) for i in range(self.N))
        return (deviance_statistic, chi2crit, df)

    def pearson_residuals(self, beta_estimate):
        muu = self.mu(self.beta)
        mu_hat = np.array([muu[i].subs(beta_estimate) for i, _ in enumerate(y)])
        r = [float((y[i] - mu_hat[i]) / mu_hat[i]**0.5) for i, _ in enumerate(y)]
        # self.plot(x=list(range(1, len(r)+1)), y=r, xlabel=r'$i$', ylabel=r'$r_i$', title='Pearson residuals')
        return mean(r), variance(r)

    def plot(self, x, y, xlabel, ylabel, title):
        fig, ax = plt.subplots()
        plt.plot(x, y, 'r')
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title)
        plt.show()

    def main(self):
        mu = self.mu(self.beta)
        logging.debug(mu)

        ll = self.log_likelihood(self.beta)
        logging.debug(ll)

        u = self.score_statistic(self.beta)
        logging.debug(u)

        beta_estimate = self.solve_for_beta(u=u, initial_guess=[0.5, 0.6, 0.7])
        logging.debug(beta_estimate)

        d, c2c, df = self.goodness_of_fit(beta_estimate)
        logging.debug(f'Deviance = {d}\nChi squared critical value = {c2c}')

        mn, var = self.pearson_residuals(beta_estimate)
        logging.debug(f'mean[r] = {mn}\nVar[r] = {var}')

if __name__ == "__main__":
    def problem():
        # Reading the group2 data from the provided file
        with open(r"group2.txt", "r") as data:
            dat = [[float(dd) for dd in d.split(",")] for d in data.readlines()]

        # Naming the data
        length, aadt, crash_count, dd = list(zip(*dat))
        vmt = (2.365+365) * np.array(aadt) * np.array(length)

        x1 = np.ones(len(vmt))
        x2 = np.log(vmt / 1e6)
        x3 = np.log(dd)

        y = np.array(crash_count)
        x = np.array(list(zip(x1, x2, x3)))

        beta = np.array([[Symbol('β1')],
                         [Symbol('β2')],
                         [Symbol('β3')]])

        return x, y, beta

    def ex_92():
        data = """1 1 65 317 2 20
1 2 65 476 5 33
1 3 52 486 4 40
1 4 310 3259 36 316
2 1 98 486 7 31
2 2 159 1004 10 81
2 3 175 1355 22 122
2 4 877 7660 102 724
3 1 41 223 5 18
3 2 117 539 7 39
3 3 137 697 16 68
3 4 477 3442 63 344
4 1 11 40 0 3
4 2 35 148 6 16
4 3 39 214 8 25
4 4 167 1019 33 114"""

        data = np.array([[int(i) for i in line.split(' ')]
                         for line in data.split('\n')])

        car = data[:, 0]
        age = data[:, 1]
        dist0 = data[:, 2:4]
        dist1 = data[:, 4:]

        return 
        
    x, y, beta = problem()
    Model = Poisson(x, y, beta)
    Model.main()
