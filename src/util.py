import zipfile

import tensorflow as tf
import numpy as np


class AverageLossStatsHelper:
    def __init__(self, frequency):
        self.average_loss = 0
        self.frequency = frequency

    def show_stats(self, loss, training_epoch_counter):
        self.average_loss += loss
        if training_epoch_counter % self.frequency == 0:
            if training_epoch_counter > 0:
                self.average_loss = self.average_loss / self.frequency
            print('Average loss at step %d: %f' % (training_epoch_counter, self.average_loss))
            self.average_loss = 0


def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions)))


def print_separator():
    print('=' * 80)
