import tensorflow as tf
import numpy as np
import src.util

from tensorflow.python.ops import nn_ops
from src.util import AverageLossStatsHelper


class LSTMModelConfig:
    def __init__(self, input_size, output_size, batch_config, num_nodes=64, input_keep_probability=0.7):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_config.batch_size
        self.num_unrollings = batch_config.num_unrollings
        self.num_nodes = num_nodes
        self.input_keep_probability = input_keep_probability


class LSTMModel:
    def __init__(self, lstm_model_config, save_to_path):
        self.lstm_model_config = lstm_model_config
        self.save_to_path = save_to_path
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.wx = tf.Variable(tf.truncated_normal([self.lstm_model_config.input_size,
                                                       4 * self.lstm_model_config.num_nodes], -0.1, 0.1))
            self.wh = tf.Variable(tf.truncated_normal([self.lstm_model_config.num_nodes,
                                                       4 * self.lstm_model_config.num_nodes], -0.1, 0.1))
            self.wb = tf.Variable(tf.truncated_normal([1, 4 * self.lstm_model_config.num_nodes], -0.1, 0.1))

            saved_output = tf.Variable(tf.zeros([self.lstm_model_config.batch_size,
                                                 self.lstm_model_config.num_nodes]),
                                       trainable=False)
            self.saved_state = tf.Variable(tf.zeros([self.lstm_model_config.batch_size,
                                                     self.lstm_model_config.num_nodes]), trainable=False)

            self.classifier_weights = tf.Variable(tf.truncated_normal([self.lstm_model_config.num_nodes,
                                                                       self.lstm_model_config.output_size],
                                                                      -0.1, 0.1))
            self.classifier_bias = tf.Variable(tf.zeros([self.lstm_model_config.output_size]))

            def lstm_cell(i, o, state):
                # Assignment 6-1
                full_matmul = tf.matmul(i, self.wx) + tf.matmul(o, self.wh) + self.wb
                input_gate, forget_gate, update, output_gate = tf.split(1, 4, full_matmul)

                input_gate = tf.sigmoid(input_gate)
                forget_gate = tf.sigmoid(forget_gate)
                output_gate = tf.sigmoid(output_gate)

                state = forget_gate * state + input_gate * tf.tanh(update)
                return output_gate * tf.tanh(state), state

            def set_optimizer(loss):
                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
                return optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

            def classify(input):
                return tf.matmul(input, self.classifier_weights) + self.classifier_bias

            self.train_data = list()
            for _ in range(self.lstm_model_config.num_unrollings + 1):
                self.train_data.append(tf.placeholder(tf.float32,
                                                      shape=[self.lstm_model_config.batch_size,
                                                             self.lstm_model_config.input_size]))

            self.train_labels = list()
            for _ in range(self.lstm_model_config.num_unrollings):
                self.train_labels.append(tf.placeholder(tf.float32,
                                                        shape=[self.lstm_model_config.batch_size,
                                                               self.lstm_model_config.output_size]))

            train_inputs = self.train_data[:self.lstm_model_config.num_unrollings]

            self.outputs = list()
            output = saved_output
            self.state = self.saved_state

            for input in train_inputs:
                # Assignment 6-3
                input = nn_ops.dropout(input, self.lstm_model_config.input_keep_probability)
                output, self.state = lstm_cell(input, output, self.state)
                self.outputs.append(output)

            with tf.control_dependencies([saved_output.assign(output),
                                          self.saved_state.assign(self.state)]):
                logits = classify(tf.concat(0, self.outputs))
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, self.train_labels)))

            self.optimizer = set_optimizer(self.loss)
            self.train_prediction = tf.nn.softmax(logits)

            self.sample_input = tf.placeholder(tf.float32, shape=[1, self.lstm_model_config.input_size])
            self.saved_sample_output = tf.Variable(tf.zeros([1, self.lstm_model_config.num_nodes]))
            self.saved_sample_state = tf.Variable(tf.zeros([1, self.lstm_model_config.num_nodes]))
            self.reset_sample_state = tf.group(
                self.saved_sample_output.assign(tf.zeros([1, self.lstm_model_config.num_nodes])),
                self.saved_sample_state.assign(tf.zeros([1, self.lstm_model_config.num_nodes]))
            )

            sample_output, sample_state = lstm_cell(self.sample_input, self.saved_sample_output,
                                                    self.saved_sample_state)

            with tf.control_dependencies([self.saved_sample_output.assign(sample_output),
                                          self.saved_sample_state.assign(sample_state)]):
                self.sample_prediction = tf.nn.softmax(classify(sample_output))

            self.saver = tf.train.Saver()

    def train(self, train_batcher,
              training_epochs_count=7001,
              summary_frequency=100):
        def set_up_training_data_feed(batches, labels):
            train_data_feed = dict()

            for i in range(self.lstm_model_config.num_unrollings + 1):
                train_data_feed[self.train_data[i]] = batches[i]

            for i in range(self.lstm_model_config.num_unrollings):
                train_data_feed[self.train_labels[i]] = labels[i]

            return train_data_feed

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            loss_stats_helper = AverageLossStatsHelper(summary_frequency)

            for training_epoch_counter in range(training_epochs_count):
                batches, labels = train_batcher.batch()
                train_data_feed = set_up_training_data_feed(batches, labels)

                _, current_loss, predictions, = session.run(
                    [self.optimizer, self.loss, self.train_prediction],
                    feed_dict=train_data_feed)

                loss_stats_helper.show_stats(current_loss, training_epoch_counter)

            self.saver.save(session, self.save_to_path)

    def generate_text(self, start_bigram, data_with_embeddings, text_samples_len=24):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, save_path=self.save_to_path)
            generated_text = []

            for bigram in start_bigram:
                bigram_embedding = data_with_embeddings.get_embedding_for_bigram(bigram)
                text_sample = bigram
                self.reset_sample_state.run()

                for _ in range(text_samples_len):
                    prediction = self.sample_prediction.eval({self.sample_input: [bigram_embedding]})
                    bigram = data_with_embeddings.get_bigram(np.argmax(prediction))
                    bigram_embedding = data_with_embeddings.get_embedding_for_bigram(bigram)
                    text_sample += bigram

                generated_text.append(text_sample)

            return generated_text

    def measure_validation_perplexity(self, data_with_embeddings):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, save_path=self.save_to_path)
            self.reset_sample_state.run()

            validation_text_size = data_with_embeddings.get_data_len()
            valid_logprob = 0

            cursor = 0
            for _ in range(validation_text_size - 1):
                predictions = self.sample_prediction.eval({self.sample_input:
                    [data_with_embeddings.get_embedding_in_position(
                        cursor)]})

                label = np.zeros(data_with_embeddings.get_bigrams_len(), dtype=float)
                label[data_with_embeddings.get_bigram_id_in_position(cursor + 1)] = 1.
                valid_logprob += src.util.logprob(predictions, label)
                cursor += 1

            return float(np.exp(valid_logprob / (validation_text_size + 0.)))
