#! python3

import numpy as np
from math import exp
import logging

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

class Sim_anneal:
    def __init__(self, function, arg_len, method='min', initial_guess=None, n=30, k=0.75, To=1000, Tf=1):
        self.n = n
        self.k = k
        self.To = To
        self.Tf = Tf
        def f(x):
            return function(*x) if method == 'min' else - function(*x)
        self.f = f
        self.xlen = arg_len
        self.initial_guess = initial_guess
            
    def is_feasible(self, x):
        """For any child class this method should be rewritten based
        on the functions feasibility"""
        return True

    def gen_xprime(self, x, neighbour=0.5):
        while True:
            x = np.random.uniform(x - neighbour, x + neighbour, size=self.xlen)
            if self.is_feasible(x):
                return x

    def sim_anneal(self):
        """Performs simulated Annealing on an optimization problem (Minimization)
        and returns the global minima."""
            
        # Step 1: Choose an initial feasible solution x from X (feasible region)
        x = np.random.uniform(*self.initial_guess, size=self.xlen)
        
        # Step 2: x* <-- x
        x_best = x

        # Step 3: T <-- To          where, To = initial highest temp
        T = self.To

        while True:
            # Step 4: Repeat the following steps with same temperature
            for iteration in range(self.n):

                # Step 4.a: Generate x' in feasible region
                x_prime = self.gen_xprime(x)
                
                # Step 4.b:
                x_best = x_prime if self.f(x_prime) <= self.f(x_best) else x_best
                
                # Step 4.c:
                if self.f(x_prime) < self.f(x):
                    x = x_prime
                # Step 4.d:
                else:
                    # As temperature T decreases probability also decreases
                    probability = exp( (self.f(x) - self.f(x_prime)) / T)
                    if probability >= np.random.uniform(0, 1):
                        x = x_prime
                
            # Step 5: Lower the Temperature
            if T > self.Tf:              # Where, Tf = final lowest temp
                T *= self.k                   #         k = multiplier (0 < k <= 1)
                
            # Step 6:
            else:
                return x_best

if __name__ == '__main__':

    def ff(x, y):
        return - (x+1)**2 - (y+2)**2 +5
    
    SA = Sim_anneal(ff, 2, 'max', initial_guess=[-10, 10], n=50, k=0.75, To=1000, Tf=1)
    x = SA.sim_anneal()
    print(x)
