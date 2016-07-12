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
        metrics: list 
    """

    def __init__(self, save_fp, metrics):
        train_loss_fp = save_fp + 'train_losses.txt'
        val_loss_fp = save_fp + 'val_losses.txt'
        self.metrics = {'val_losses': [], 'train_losses': []}
        self.filepaths = {'train_losses': train_loss_fp, 
                          'val_losses': val_loss_fp}
        
        # ensure the directorys where metrics are stored are created
        if not os.path.exists(os.path.dirname(train_loss_fp)):
            os.makedirs(os.path.dirname(train_loss_fp), exist_ok=True)

        if 'accuracy' in metrics:
            train_acc_fp = save_fp + 'train_accs.txt'
            val_acc_fp = save_fp + 'val_accs.txt'
            
            self.metrics['train_accs'], self.metrics['val_accs'] = [], []
            self.filepaths['train_accs'] = train_acc_fp
            self.filepaths['val_accs'] = val_acc_fp

    def on_batch_end(self, batch, logs):
        """Append the training loss to `train_losses`.

        Args:
        ----
            batch: int
            logs: dict
        """

        self.metrics['train_losses'].append(logs['loss'])
        
        if 'acc' in logs:
            self.metrics['train_accs'].append(logs['acc'])

    def on_epoch_end(self, epoch, logs): 
        """Save the losses from the batches/epoch to the appropriate filepaths.

        Args: 
        ----
            epoch: int
            logs: dict
        """

        if 'val_loss' in logs: 
            self.metrics['val_losses'].append(logs['val_loss'])

        if 'val_acc' in logs: 
            self.metrics['val_accs'].append(logs['val_acc'])
        
        for name, metric_lst in self.metrics.items():
            if not metric_lst: 
                continue 
            metric_fp = self.filepaths[name]
            with open(metric_fp, 'a+') as f: 
                for metric in metric_lst: 
                    f.write(str(metric))
                    f.write('\n')
            self.metrics[name] = []
