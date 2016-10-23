import collections
import random

import numpy as np


class BatcherConfig:
    def __init__(self, batch_size, num_skips, skip_window):
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.batch_size = batch_size


class Batcher:
    def __init__(self, processed_data, batch_config):
        self.data = processed_data
        self.cursor = 0
        self.batch_config = batch_config

    def get_batch_size(self):
        return self.batch_config.batch_size

    def batch(self):
        """

        :param batch_size:
        :param num_skips: how many elements from window take into account
        :param skip_window: how many elements put into window in one side. all window length 2*skip_window + 1
        :return:
        """
        assert self.batch_config.batch_size % self.batch_config.num_skips == 0
        assert self.batch_config.num_skips <= 2 * self.batch_config.skip_window

        batch = np.ndarray(shape=self.batch_config.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_config.batch_size, 1), dtype=np.int32)

        span = 2 * self.batch_config.skip_window + 1
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            self._add_next_data_to_buffer(buffer)

        for i in range(self.batch_config.batch_size // self.batch_config.num_skips):
            target = self.batch_config.skip_window
            targets_to_avoid = [self.batch_config.skip_window]

            for j in range(self.batch_config.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)

                targets_to_avoid.append(target)
                batch[i * self.batch_config.num_skips + j] = buffer[self.batch_config.skip_window]
                labels[i * self.batch_config.num_skips + j, 0] = buffer[target]

            self._add_next_data_to_buffer(buffer)

        return batch, labels

    def _add_next_data_to_buffer(self, buffer):
        buffer.append(self.data.text_from_bigrams[self.cursor])
        self.cursor = (self.cursor + 1) % self.data.text_from_bigrams_len
