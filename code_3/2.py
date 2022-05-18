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

logging.disable(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class GLM():
    def __init__(self, ):
        
    def mu(beta, x):
        return np.array([sympy.exp(xb[0]) for xb in x.dot(beta)])

    def log_likelihood(beta, x):
        x_dot_beta = x.dot(beta)
        muu = mu(beta, x)
        y_facttorials = [float(factorial(int(yy))) for yy in y]
        ll = sum( - muu[i] + y[i]*x_dot_beta[i] - np.log(y_facttorials[i])
                    for i, _ in enumerate(y))[0]
        return ll

    def score_statistic(beta, x):
        muu = mu(beta, x)
        u = [sum(x[i, j]*(y[i] - muu[i]) for i, _ in enumerate(y)) for j in range(x.shape[1])]
        
        return u

    def solve_for_beta(initial_guess=None, max_iter=20):
        betas = [b[0] for b in beta]
        i = 0
        while i < max_iter:
            initial_guess = [np.random.uniform(-1, 1) for _ in beta] if not initial_guess else initial_guess
            
            system_of_nonlinear_eqns = score_statistic(beta, x)

            logging.info('\n\nTo maximize likelihood function we need to set all the score statistics equal to zero and'
                         ' solve the system of equations.')
            for eqn in system_of_nonlinear_eqns:
                print(f'\n{eqn} = 0\n')
            
            logging.info(f"\n\nUsing the initial guess = {initial_guess} as the beta vector")

            try:
                soln = sympy.nsolve(system_of_nonlinear_eqns, betas, initial_guess)
                break
            except ValueError:
                logging.info("BAD initial guess, changing the guess.")
                initial_guess = None
            i += 1
            
        soln = {b: br for b, br in zip(betas, soln)}
        print(soln)
        
        return soln
        
    def goodness_of_fit():
        df = number_of_datapoints - len(beta)
        chi2crit = chi2.ppf(0.95, df)
        print(f'\nFor 95% confidence interval with df = {number_of_datapoints}'
              f' - {len(beta)} = {df}\nChi Squared Critical Value = {chi2crit}\n')

        deviance_statistic = 2 * sum((- (y[i] - mu_hat[i])) + (y[i] * np.log(float(y[i]/mu_hat[i]))
                                                    if y[i] > 0 else 0) for i, _ in enumerate(y))

        sign, fit_type = (">>", "poor") if deviance_statistic > chi2crit else ("<<", "good")
        
        print(f'Deviance statistics, {deviance_statistic} {sign} Chi Squared Critical Value, '
              f'{chi2crit}.\nSo, We have a {fit_type} fit')

    def standardized_residuals():
        r = [float((y[i] - mu_hat[i]) / mu_hat[i]**0.5) for i, _ in enumerate(y)]
        m = mean(r)
        v = variance(r)
        print(f'mean(r) = {m}\nvar(r) = {v}\nsqrt(var(r)) = {v**0.5}\n')
        print("For poisson distribution, mean(r) = var(r).\n"
              f"Here, mean(r) = {m} {'<<' if m<v else ('=' if m==v else '>')} var(r) = {v}.")
        print(f"\nSo, Model is{'' if m<v else ' not'} overdispersed.")
        plot(x=list(range(1, len(r)+1)), y=r, xlabel=r'$i$', ylabel=r'$r_i$', title='Pearson residuals')
        return r

    def lawless_residual():
        y_bar = mean(y)
        s = sum((y[i] - mu_hat[i])**2 - y_bar for i, _ in enumerate(y)) / (2 * sum(mu_hat[i]**2 for i, _ in enumerate(y)))**0.5
        print(f'Standardized dispersion statistic from Lawless\'s paper, S = {s}')
        return s

    def plot(x, y, xlabel, ylabel, title):
        fig, ax = plt.subplots()
        plt.plot(x, y, 'r')
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title)
        plt.show()


if __name__ == "__main__":

    def data():
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

    x, y, beta = data()
    
    number_of_datapoints = len(y)

    
    beta_result = solve_for_beta()
    beta_result_np = np.array(list(beta_result.values()))
    mu_hat = np.array([mu(beta, x)[i].subs(beta_result) for i, _ in enumerate(y)])
    goodness_of_fit()
    ##r = standardized_residuals()
    s = lawless_residual()
