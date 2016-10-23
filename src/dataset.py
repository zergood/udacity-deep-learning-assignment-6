import numpy as np


class Dataset:
    @staticmethod
    def create_from_text(text):
        """
        :param text - initial text to process
        :return:
         bigram2id - mapping bigram value to bigram id(index)
         text_from_bigrams - initial text where letters is substituted by appropriate bigram id
         bigrams - bigrams array
        """
        bigrams2id = {}
        bigrams = []
        text_from_bigrams = []

        cursor = 0
        while cursor < len(text) - 1:
            bigram = text[cursor] + text[cursor + 1]
            if bigram not in bigrams2id:
                bigram_id = len(bigrams2id)
                bigrams2id[bigram] = bigram_id
                bigrams.append(bigram)
                text_from_bigrams.append(bigram_id)
            else:
                text_from_bigrams.append(bigrams2id[bigram])

            cursor += 2

        assert len(bigrams) == len(bigrams2id)
        return Dataset(bigrams2id, text_from_bigrams, bigrams)

    def __init__(self, bigrams2id, text_from_bigrams, bigrams):
        self.bigrams2id = bigrams2id
        self.text_from_bigrams = text_from_bigrams
        self.bigrams = bigrams
        self.text_from_bigrams_len = len(self.text_from_bigrams)

    def train_validation_split(self, validation_size):
        """

        :param validation_size:
        :return: train and validation datasets
        """
        assert validation_size >= self.get_bigrams_len()

        validation_dataset = Dataset(self.bigrams2id, self.text_from_bigrams[-validation_size:], self.bigrams)
        train_dataset = Dataset(self.bigrams2id, self.text_from_bigrams[:-validation_size], self.bigrams)
        return train_dataset, validation_dataset

    def get_bigram(self, index):
        return self.bigrams[index]

    def get_bigram_id(self, bigram):
        return self.bigrams2id[bigram]

    def get_bigrams_len(self):
        return len(self.bigrams)


class DatasetWithEmbeddings:
    def __init__(self, dataset, embeddings):
        self.processed_data = dataset
        self.embeddings = embeddings

    def get_embedding_in_position(self, position):
        bigram_id = self.processed_data.text_from_bigrams[position]
        return self.embeddings[bigram_id]

    def get_bigram_id_in_position(self, position):
        return self.processed_data.text_from_bigrams[position]

    def get_data_len(self):
        return len(self.processed_data.text_from_bigrams)

    def get_embedding_size(self):
        return self.embeddings.shape[1]

    def get_bigram_from_embedding(self, embedding):
        embedding_np = np.array(embedding)
        current_dist = np.inf
        current_closest = 0
        for i in range(len(self.embeddings)):
            dist = np.linalg.norm(embedding_np - self.embeddings[i])
            if dist < current_dist:
                current_closest = i
                current_dist = dist
        return self.processed_data.get_bigram(current_closest)

    def get_embedding_for_bigram(self, bigram):
        id = self.processed_data.get_bigram_id(bigram)
        return self.embeddings[id]

    def get_bigram(self, id):
        return self.processed_data.get_bigram(id)

    def get_bigrams_len(self):
        return self.processed_data.get_bigrams_len()
