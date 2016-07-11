"""A script for Keras callbacks."""

import os
from keras.callbacks import Callback

class LossSaver(Callback):
    """A utility class for saving losses per batch and epoch.

    This class keeps track of the loss for each batch, and saves those losses
    to disk at the end of each epoch.

    Args: 
    ----
        save_fp: str
    """

    def __init__(self, save_fp):
        self.train_fp = save_fp + 'train_losses.txt'
        self.val_fp = save_fp + 'val_losses.txt'

        if not os.path.exists(os.path.dirname(self.train_fp)):
            os.makedirs(os.path.dirname(self.train_fp), exist_ok=True)
            
        self.train_losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs):
        """Append the training loss to `train_losses`.

        Args:
        ----
            batch: int
            logs: dict
        """

        self.train_losses.append(logs['loss'])

    def on_epoch_end(self, epoch, logs): 
        """Save the losses from the batches/epoch to the appropriate filepaths.

        Args: 
        ----
            epoch: int
            logs: dict
        """

        if 'val_loss' in logs: 
            self.val_losses.append(logs['val_loss'])

        with open(self.train_fp, 'a+') as f: 
            for loss in self.train_losses: 
                f.write(str(loss))
                f.write('\n')
            self.train_losses = []

        if self.val_losses: 
            with open(self.val_fp, 'a+') as f: 
                for loss in self.val_losses: 
                    f.write(str(loss))
                    f.write('\n')
                self.val_losses = []
