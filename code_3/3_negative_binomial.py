#! python3

import numpy as np
import os
import logging
from sympy import Symbol
from sympy.solvers import solve
import sympy
from math import factorial, lgamma
from scipy.stats import chi2
from scipy.optimize import bisect
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from statistics import mean, variance
from mpmath import findroot
from simulated_annealing import Sim_anneal

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

    def floater(self, x):
        try:
            return float(x)
        except:
            return x[0]
    
    def mu(self, beta):
        nbeta = np.array(list(beta.values())) if type(beta) == dict else beta
        self.__mu = np.array([sympy.exp(self.floater(xb)) for xb in self.x.dot(nbeta)])
        return self.__mu

    def lgm(self, a, y):
        return 0 if y <= 0 else sum( np.log(1+a*j) for j in range(0, int(y))) # range(start, end-1)
    
    def log_likelihood_lp(self, a, beta_estimate):
        muu = self.mu(self.beta)
        mu_e = np.array([muu[i].subs(beta_estimate) for i, _ in enumerate(y)])
        return sum(self.lgm(a, y[i]) + y[i]*mu_e[i] - (y[i] + 1/a) * np.log(float(1+a*mu_e[i])) for i in range(self.N))
        
    def log_likelihood_1(self, beta):
        y_factorials = [float(factorial(int(yy))) for yy in y]
        nbeta = np.array(list(beta.values())) if type(beta) == dict else beta
        x_dot_beta = x.dot(nbeta)
        muu = self.mu(beta)
        result = sum(-muu[i] + y[i]*x_dot_beta[i] - np.log(y_factorials[i]) for i in range(self.N))
        return self.floater(result)

    def log_likelihood(self, a, beta):
        return self.log_likelihood_1(beta) if a<=0 else self.log_likelihood_lp(a, beta)

    def __ll(self, beta):
        def nll(a):
            return self.log_likelihood(a, beta)
        return nll

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

    def method_of_moments(self, beta_estimate, initial_guess):
        muu = self.mu(self.beta)
        y_hat = np.array([muu[i].subs(beta_estimate) for i, _ in enumerate(y)])
        a = Symbol('a')
        f = sum((y[i] - y_hat[i])**2 / (y_hat[i] * (1 + a*y_hat[i])) for i, _ in enumerate(y)) - (self.N - self.p)
        return sympy.nsolve(f, a, initial_guess)

    def negative_binomial(self, a):
        muu = self.mu(self.beta)
        y_hat = np.array([muu[i] for i, _ in enumerate(y)])
        return [sum(self.x[i, j]*(self.y[i] - y_hat[i])/(1 + a*y_hat[i])
                    for i in range(self.N)) for j in range(self.xc)]

    def get_dispersion_param_iterative(self, initial_guess, beta_estimate):
        log_likelihood_func = self.__ll(beta_estimate)    
        a_min = initial_guess - 0.05
        a_max = initial_guess + 0.05
        
##        Solver = Sim_anneal(log_likelihood_func, 1, 'max', initial_guess=[a_min, a_max], n=30, k=0.75, To=2, Tf=1)
##        a = Solver.sim_anneal()[0]
##        print(f'Simulated anealing a = {a}')

        # Brute Force
        _, a = max((log_likelihood_func(float(a)), a) for a in np.arange(a_min, a_max, 0.001))
        
        print(a)
        
        return a
    
    def estimates(self):
        mu = self.mu(self.beta)
##        logging.debug(mu)

        ll = self.log_likelihood_1(self.beta)
##        logging.debug(ll)

        u = self.score_statistic(self.beta)
##        logging.debug(u)

        beta_estimate = self.solve_for_beta(u=u, initial_guess=[0.5, 0.6, 0.7])
        logging.debug(beta_estimate)

        d, c2c, df = self.goodness_of_fit(beta_estimate)
        logging.debug(f'Deviance = {d}\nChi squared critical value = {c2c}')

        mn, var = self.pearson_residuals(beta_estimate)
        logging.debug(f'mean[r] = {mn}\nVar[r] = {var}')

        a = self.method_of_moments(beta_estimate, 0.3)
        logging.debug(a)

        neg_bin_u = self.negative_binomial(a)
        beta_new = self.solve_for_beta(neg_bin_u, list(beta_estimate.values()))

        a_new = self.get_dispersion_param_iterative(a, beta_new)
        neg_bin_u_2 = self.negative_binomial(a)
        beta_newer = self.solve_for_beta(neg_bin_u_2, list(beta_new.values()))

        return beta_newer
        
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

    x, y, beta = problem()
    Model = Poisson(x, y, beta)
    betas = Model.estimates()
    print(betas)
