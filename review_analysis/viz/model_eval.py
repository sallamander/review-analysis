"""A script for plotting the results of models."""

import numpy as np
import matplotlib.pyplot as plt

class NetErrorPlotter(object): 
    """A class for plotting errors from neural network training."""

    def __init__(self, errors): 
        self.errors = errors 
        self._build_plot()

    def _build_plot(self): 
        
        num_subplots = len(self.errors)
        num_rows, num_cols = self._calc_subplot_dims(num_subplots)
        self.fig, self.axes = plt.subplots(num_rows, num_cols)
        self._calc_bounds()
        for ax, errors_tup in zip(self.axes.flatten(), self.errors):
            self._plot_subplot(ax, errors_tup)

    def _calc_subplot_dims(self, num_subplots):
        """A tiny utility for calculating subplot dimensions. 

        Calculate the subplot dimensions (e.g. # rows, and # cols) from the 
        inputted total `num_subplots`. In order to make sure there are the 
        minimum number of subplots generated: 
            
            1. Calc the square root of `num_subplots`, and floor it. 
            2. Check to see if (square_floor + 1, square_floor) will be enough
               subplots
            3. If not, then use (square_floor + 1, square_floor + 1)
        
        This ensures that 40 subplots would give a 6 * 7 and 45 subplots would
        give a 7 * 7 (without step 3 this would give a 6 * 7 for both). 

        Args: 
        ----
            num_subplots: int

        Return:
        ------
            (int, int)
        """

        square_floor = int(num_subplots ** 0.5)
        num_plots_w_floor = square_floor * (square_floor + 1)

        if num_plots_w_floor < num_subplots: 
            return (square_floor + 1, square_floor + 1)
        else: 
            return (square_floor + 1, square_floor)

    def _calc_bounds(self):
        """Calc the global y min/max over errors to standarize subplot axes."""
        
        mins, maxes = [], []
        for _, errors, train in self.errors: 
            # Training has errors by batch - skip the first 1000 to avoid noise
            skip = 1000 if train else 0
            mins.append(errors[skip:].min())
            maxes.append(errors[skip:].max())

        self.min = np.min(mins)
        self.max = np.max(maxes)

    def _plot_subplot(self, ax, error_tup):
        """Plot the inputted `errors` on the inputted `ax`.

        Args:
        ----
            ax: matplotlib.pyplot.Axes object
            error_tup: dict
                holds `title` (str), `errors` (1d np.ndarray) pair
        """
        
        title = error_tup[0]
        errors = error_tup[1]
        
        ax.plot(errors)
        ax.set_ylim(self.min, self.max)
        ax.set_title(title)

    def show(self): 
        """Show the plot."""

        plt.show()

if __name__ == '__main__':
    paths = ['work/mean_squared_error/GRU/adagrad_8_0_train_losses.txt', 
             'work/mean_squared_error/GRU/rmsprop_8_0_train_losses.txt']
    
    titles = ['GRU - Adagrad - 8', 'GRU - RMSProp - 8']

    errors_lst = []
    for title, path in zip(titles, paths):
        errors = np.loadtxt(path)
        errors_lst.append((title, errors, True))

    plotter = NetErrorPlotter(errors_lst)
    plt.show()
    plotter.show()
