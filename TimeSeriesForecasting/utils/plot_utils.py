from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np


def ts_animation(y_te, y_hat, n_rows=50, y_min=-2, y_max=2):
    "plot the first n_rows of the two y_te and y_hat matrices"
    fig, ax = plt.subplots(1);

    line1, = ax.plot(y_hat[0], lw=2);
    line2, = ax.plot(y_hat[0], lw=2);
    ax.set_ylim(y_min, y_max)
    n_sa = y_hat.shape[1]
    def animate(i):
        line1.set_data(np.arange(n_sa), y_te.values[i:i + n_sa]);
        line2.set_data(np.arange(n_sa), y_hat[i, :]);
        return (line1, line2)

    def init():
        line1.set_data([], []);
        return (line1,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_rows, interval=100,
                                  save_count=50, blit=True)
    plt.close('all')
    # rc('animation', html='jshtml')
    return HTML(ani.to_jshtml())