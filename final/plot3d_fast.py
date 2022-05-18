import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np

def plot3d_fast(f, x=(0, 5, 'time, t (x-axis)'), y=(0, 5, 'space, x (y-axis)')):        
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    # plot a 3D surface
    X = np.arange(x[0], x[1], 0.25)
    T = np.arange(y[0], y[1], 0.25)
    T, X = np.meshgrid(T, X)
    Z = f(X,T)
    
    surf = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True)

    fig.colorbar(surf, shrink=0.5, aspect=10)
    titles = {'N': "Cumulative Count", 'k': 'Density', 'q': 'Flow'}
    plt.title(titles.get(f.__name__, ''))
    ax.set_xlabel(x[2], fontsize=14)
    ax.set_ylabel(y[2], fontsize=14)
    ax.set_zlabel(f"{titles[f.__name__]}, {f.__name__}", fontsize=14)
##    ax.set_xlim(*x)
##    ax.set_ylim(*y)
    
##    plt.zlabel("N(x,t)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    def N(x, t):
        return 60*t - 120*x + (60*x**2)/(t+1)
    def k(x, t):
        return 120*(1-(x/(t+1)))
    def q(x, t):
        return 60*(1 - (x / (t+1))**2)
    plot3d_fast(N, x=(0, 5, ''), y=(0, 5, ''))
