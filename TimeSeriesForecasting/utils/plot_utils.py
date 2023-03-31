from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np


import seaborn as sns
import matplotlib.gridspec as gridspec


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



class SeabornFig2Grid():
    """
    Plot multiple distplots in different subplots.
    Move a seaborn FacetGrid or PairGrid to a GridSpec within a figure.
    """

    def __init__(self, seaborngrid, labels, figsize=(13, 8)):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, len(seaborngrid))

        for sbg, s, t in zip(seaborngrid, gs, labels):
            self.fig = fig
            self.sg = sbg
            self.subplot = s
            if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
                self._movegrid()
            elif isinstance(self.sg, sns.axisgrid.JointGrid):
                self._movejointgrid()
            self._finalize()
            sbg.ax_marg_x.set_title(t)
            plt.show()
        gs.tight_layout(fig)



    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
