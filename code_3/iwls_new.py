#! python3

import numpy as np
import os
import logging

logging.disable(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class IWLS:
    def __init__(self, x, y):

##        self.x = np.array([[1, i] for i in x])
        self.x = np.array(x)
        self.y = np.array(y).reshape(-1, 1)

        self.alt_initial_guess = np.random.randint(1e6, size=self.x.shape[1])

        assert len(x) == len(y), "Dataset incomplete" 
        self.N = len(y)
    
    def run_iwls(self, initial_guess, epsilon=1e-5, max_iter=10):

        # Ensuring b vector's dimension in (nx1) format
        b = np.array(initial_guess if initial_guess else self.alt_initial_guess).reshape(-1, 1)

        # Start of algorithm IWLS:
        for iteration in range(1, max_iter+1):

            # Need to keep track of old b to compare abs(b - b_old) with epsilon
            b_old = b
            
            # Getting eta
##            n = [x_i.dot(b)[0] for x_i in self.x]
            n = self.x.dot(b.reshape(-1, 1))

            # Setting mu equal to eta
            mu = n

            # Calculating the vector z
##            z = np.array([n[i] + (self.y[i] - mu[i]) for i in range(self.N)])
            z = n + self.y - mu
            
            # Getting all the w_ii values
            wii = np.array([1/mu_i for mu_i in mu])

            # Creating the diagonal matrix W using w_ii
            w = np.diag(wii.flatten())

            # Calculating the information matrix, j = x^T . w. x
            j = self.x.transpose().dot(w.dot(self.x))

            # Calculating v = x^T . w. z
            v = self.x.transpose().dot(w.dot(z))

            # Solving for beta vector
            b = np.linalg.inv(j).dot(v)

            logging.debug(f"mu = \n{np.array(mu).reshape(-1, 1)}\n\nz = \n{z}\n\nb = \n{b}\n\n"
                         f"w = \n{w}\n\nj = \n{j}\n\nv = \n{v}")

            print(f"\nAfter Iteration-{iteration}:\nBeta = \n{b}\n")

            # Stopping criteria check
            if all(abs(b-b_old) < epsilon):
                break
            
        return b

    def a_cov_matrix(self, b):
        n = self.x.dot(b.reshape(-1, 1))

        # Setting mu equal to eta
        mu = n
        
        # Getting all the w_ii values
        wii = np.array([1/mu_i for mu_i in mu])

        # Creating the diagonal matrix W using w_ii
        w = np.diag(wii.flatten())
        
        # Calculating the information matrix, j = x^T . w. x
        j = self.x.transpose().dot(w.dot(self.x))

        return w, j
        
if __name__ == "__main__":
    def example_1():
        x = [-1, -1, 0, 0, 0, 0, 1, 1, 1]
        y = [2, 3, 6, 7, 8, 9, 10, 12, 15]
        return x, y

    def db_4_2():
        x = [3.36, 2.88, 3.63, 3.41, 3.78, 4.02, 4.00, 4.23, 3.73, 3.85, 3.97, 4.51, 4.54, 5.00, 5.00, 4.72, 5.00]
        y = [65, 156, 100, 134, 16, 108, 121, 4, 39, 143, 56, 26, 22, 1, 1, 5, 65]
        exp_y = [np.log(y_i) for y_i in y]
        return x, exp_y

    def problem_test():
        with open(r"group2.txt", "r") as data:
            dat = [[float(dd) for dd in d.split(",")] for d in data.readlines()]
        y, *x = list(zip(*dat))
        x = list(zip(*x))

        return x, y

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

##        beta = np.array([[Symbol('β1')],
##                         [Symbol('β2')],
##                         [Symbol('β3')]])

        return x, y

    x, y = problem()
    
    
##    Solver = IWLS(*db_4_2())
    Solver = IWLS(x, y)
    # Using beta_1 = 7, beta_2 = 5 as initial guess 
    beta = Solver.run_iwls(initial_guess=[0.5, 0.6, 0.7], max_iter=20)
    w, j = Solver.a_cov_matrix(beta)
    print(w)
    print(j)
    
