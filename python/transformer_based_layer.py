import keras
import keras_nlp
import tensorflow as tf


class TransformerBasedLayer(keras.layers.Layer):
    def __init__(self, n_heads, d_model, d_feed_forward, d_out, n_bidders, n_items):
        super(TransformerBasedLayer, self).__init__()
        self.n_bidders = n_bidders + 1
        self.n_items = n_items
        self.d_model = d_model
        self.rowTransformer = keras_nlp.layers.TransformerEncoder(d_feed_forward, n_heads)
        self.columnTransformer = keras_nlp.layers.TransformerEncoder(d_feed_forward, n_heads)
        self.conv1 = keras.layers.Conv2D(d_feed_forward, (1, 1), activation=keras.activations.relu)
        self.conv2 = keras.layers.Conv2D(d_out, (1, 1), activation=keras.activations.linear)

    def call(self, inputs):

        batch_size = inputs.shape[0]

        rows = tf.reshape(inputs, (-1, self.n_items, self.d_model))
        rows = self.rowTransformer(rows)
        rows = tf.reshape(rows, (batch_size, self.n_bidders, self.n_items, -1))

        columns = tf.reshape(tf.transpose(inputs, perm=[0, 2, 1, 3]), (-1, self.n_bidders, self.d_model))
        columns = self.columnTransformer(columns)
        columns = tf.reshape(columns, (batch_size, self.n_items, self.n_bidders, -1))
        columns = tf.transpose(columns, perm=[0, 2, 1, 3])

        global_mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        global_mean = tf.tile(global_mean, [1, self.n_bidders, self.n_items, 1])

        out = tf.concat([rows, columns, global_mean], axis=-1)
        out = self.conv1(out)
        out = self.conv2(out)

        return out
