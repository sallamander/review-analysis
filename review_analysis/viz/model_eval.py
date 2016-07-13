"""A script for plotting the results of models."""

import numpy as np
import matplotlib.pyplot as plt

class NetErrorPlotter(object): 
    """A class for plotting errors from neural network training.

    Args:
    -----
        errors: list of tuples
            tuples each represent a model run and are expected to contain 
            (str, 1d np.ndarray, bool) that represent (title, errors, training)
    """

    def __init__(self, errors): 
        self.errors = errors 
        self._build_plot()

    def _build_plot(self): 
        """Build the plot with the inputted errors."""
        
        num_subplots = len(self.errors)
        num_rows, num_cols = self._calc_subplot_dims(num_subplots)
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
        if num_subplots == 1: 
            self.axes = np.array([self.axes]) # Ensure the flatten below works.
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

        if num_subplots == 1: 
            return (1, 1)

        square_floor = int(num_subplots ** 0.5)
        num_plots_w_floor = square_floor * (square_floor + 1)

        if num_plots_w_floor < num_subplots: 
            return (square_floor + 1, square_floor + 1)
        elif square_floor ** 2 == num_subplots:
            return (square_floor, square_floor)
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

        self.ymin = 0

        ymax = np.max(maxes)
        self.ymax = ymax + ymax * 0.10 # give the top of the plot a little room

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
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Loss')
        ax.set_title(title)

def gen_title(path): 
    """Generate a title string from the inputted path.

    Each inputted string path is expected to take a specific format - any number
    directories, followed by: 
    'CELL/<optimizer>_<encoding_size>_<dropout_pct>_<train/test>_<losses_accs>.txt'

    Args: 
    ----
        path: str

    Return: 
    ------
        title: str
    """

    all_parts = path.split('/')
    options_parts = all_parts[-1].split('_')
    options_parts[-1] = options_parts[-1][:-4] # Strip the .txt

    cell = all_parts[-2]
    title = cell
    for idx, part in enumerate(options_parts):
        title += ' - ' + part
    
    # make it a little prettier
    replacements = (('adagrad', 'Adagrad'), ('rmsprop', 'RMSProp'), 
                    ('train', 'Train'), ('val', 'Val'))
    for replacement in replacements: 
        title = title.replace(replacement[0], replacement[1])
    
    return title

if __name__ == '__main__':
    base_path = 'work/mean_squared_error/'
    unit_types = ('LSTM', )
    encoding_sizes = (8, 16, 32, 64)
    optimizers = ('adagrad', 'rmsprop')
    
    paths = []
    for unit_type in unit_types:
        for optimizer in optimizers: 
            for encoding_size in encoding_sizes: 
                path = base_path + '{}/{}_{}_0_train_losses.txt'.format(unit_type, 
                                                                        optimizer, 
                                                                        encoding_size)
                paths.append(path) 

    errors_lst = []
    for path in paths:
        errors = np.loadtxt(path)
        title = gen_title(path)
        training = True if 'Train' in title else False 
        errors_lst.append((title, errors, training))

    plotter = NetErrorPlotter(errors_lst)
    plt.tight_layout()
    plotter.fig.savefig('work/viz/{}_train.png'.format(unit_types[0]))
