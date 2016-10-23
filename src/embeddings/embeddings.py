import math
import tensorflow as tf

from src.util import AverageLossStatsHelper


class Embeddings:
    def __init__(self, batcher, embedding_length, vocabulary_size, num_sampled):
        self.batcher = batcher
        self.embedding_length = embedding_length
        self.num_sampled = num_sampled
        self.vocabulary_size = vocabulary_size

        self.graph = tf.Graph()

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.train_dataset = tf.placeholder(tf.int32, shape=[batcher.get_batch_size()])
            self.train_labels = tf.placeholder(tf.int32, shape=[batcher.get_batch_size(), 1])

            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_length], -1.0, 1.0))
            softmax_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_length],
                                    stddev=1.0 / math.sqrt(embedding_length)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)

            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                           self.train_labels, num_sampled, vocabulary_size)
            )

            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

            self.normalized_embeddings = embeddings / norm

    def build_embeddings(self, training_epochs_count, stats_frequency=2000):
        loss_stats_helper = AverageLossStatsHelper(stats_frequency)

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()

            for training_epoch_counter in range(training_epochs_count):
                batch_data, batch_labels = self.batcher.batch()
                _, current_loss = session.run([self.optimizer, self.loss],
                                              feed_dict={self.train_dataset: batch_data,
                                                         self.train_labels: batch_labels})

                loss_stats_helper.show_stats(loss=current_loss, training_epoch_counter=training_epoch_counter)

            return self.normalized_embeddings.eval()
