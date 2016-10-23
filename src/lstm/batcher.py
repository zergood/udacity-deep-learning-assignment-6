import numpy as np

#TODO: rename batch generator to bathes
class BatchConfig:
    def __init__(self, batch_size, num_unrollings):
        self.num_unrollings = num_unrollings
        self.batch_size = batch_size


class BatchGenerator(object):
    def __init__(self, data_with_embeddings, batch_config):
        self._data = data_with_embeddings
        self.data_size = data_with_embeddings.get_data_len()
        self.embedding_size = data_with_embeddings.get_embedding_size()
        self.bigrams_size = data_with_embeddings.get_bigrams_len()
        self.batch_size = batch_config.batch_size
        self.num_unrollings = batch_config.num_unrollings

        segment = self.data_size // batch_config.batch_size
        self._cursor = [offset * segment for offset in range(batch_config.batch_size)]
        self._last_batch, self._last_labels = self._next_mini_batch()

    def _next_mini_batch(self):
        batch = np.zeros(shape=(self.batch_size, self.embedding_size), dtype=np.float)
        labels = np.zeros(shape=(self.batch_size, self.bigrams_size), dtype=np.float)

        for b in range(self.batch_size):
            batch[b] = self._data.get_embedding_in_position(self._cursor[b])
            bigram_id = self._data.get_bigram_id_in_position(self._cursor[b])
            labels[b, bigram_id] = 1.
            self._cursor[b] = (self._cursor[b] + 1) % self.data_size
        return batch, labels

    def batch(self):
        batches = [self._last_batch]
        labels = [self._last_labels]

        for step in range(self.num_unrollings):
            batch, mini_batch_labels = self._next_mini_batch()
            batches.append(batch)
            labels.append(mini_batch_labels)

        self._last_batch = batches[-1]
        self._last_labels = labels[-1]

        return batches, labels[1:]


def batches2string(batches, labels, data_with_engrams):
    batches_string = [''] * batches[0].shape[0]
    for mini_batch in batches:
        for i in range(len(mini_batch)):
            batches_string[i] += data_with_engrams.get_bigram_from_embedding(mini_batch[i])

    labels_string = [''] * labels[0].shape[0]
    for labels_mini_batch in labels:
        for i in range(len(labels_mini_batch)):
            labels_string[i] += data_with_engrams.get_bigram(np.argmax(labels_mini_batch[i]))

    return batches_string, labels_string
