import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import keras
from keras import Input, Model
from keras.src.layers import Conv2D
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from big_layer import BigLayer
from transformer_based_layer import TransformerBasedLayer


def negative_payment_loss(y_true, y_pred):
    return tf.negative(tf.reduce_mean(tf.reduce_sum(y_pred, axis=0), axis=0))


def payment_metric(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(y_pred, axis=0), axis=0)


tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


def generate_values(samples, n_bidders, n_items, dx, dy):
    x = tf.concat([tf.random.uniform((samples, n_bidders, dx), minval=-1.0, maxval=1.0),
                   tf.tile(tf.expand_dims(tf.expand_dims(tf.repeat(1.0, dx), axis=0), axis=0), [samples, 1, 1])]
                  , axis=1)
    y = tf.random.uniform((samples, n_items, dy), minval=-1.0, maxval=1.0)

    x_exp1 = tf.expand_dims(x[:, :n_bidders, :], axis=2)  # (samples, n_bidders, 1, dx)
    y_exp = tf.expand_dims(y, axis=1)  # (samples, 1, n_items, dy)
    upper = tf.sigmoid(tf.reduce_sum(x_exp1 * y_exp, axis=-1))
    bids = tf.random.uniform((samples, n_bidders, n_items)) * upper

    x_exp2 = tf.expand_dims(x, axis=2)
    batch_data = tf.concat([tf.tile(x_exp2, [1, 1, n_items, 1]), tf.tile(y_exp, [1, n_bidders + 1, 1, 1])], axis=-1)

    return batch_data, bids


import argparse


def main():
    args, __ = parse_args()
    batch_size = 2048  # Make sure this is a divisor of samples
    epochs = args.train_steps  # train_steps
    samples = 32768  # train_sample_num = eval_sample_num
    seed = args.seed

    start_lr = 1e-8  # This is static in the code
    end_lr = 3e-4  # lr
    lr_fn = keras.optimizers.schedules.PolynomialDecay(start_lr, 100, end_lr, power=1.0)

    n_bidders = args.n_agents
    n_items = args.m_items  # m_items
    dx = 10
    dy = dx
    d_hidden = 64
    d = d_hidden
    interim_d = d
    n_interaction = 3  # n_layer

    n_heads = 4  # n_head
    menu_size = args.menu_size  # menu_size
    d_out = 2 * menu_size + 1

    temp = 500.0  # init_softmax_temperature
    allocation_temp = args.alloc_softmax_temperature  # alloc_softmax_temperature

    keras.utils.set_random_seed(seed)

    transformer_layers = []
    for i in range(n_interaction):
        if i == 0:
            transformer_layers.append(
                TransformerBasedLayer(n_heads, d, d_hidden, d_hidden, n_bidders, n_items, batch_size))
        elif i == n_interaction - 1:
            transformer_layers.append(
                TransformerBasedLayer(n_heads, d_hidden, d_hidden, d_out, n_bidders, n_items, batch_size))
        else:
            transformer_layers.append(
                TransformerBasedLayer(n_heads, d_hidden, d_hidden, d_hidden, n_bidders, n_items, batch_size))

    representations_input = Input(shape=(n_bidders + 1, n_items, dx + dy), batch_size=batch_size)
    bids_input = Input(shape=(n_bidders, n_items), batch_size=batch_size)

    conv1 = Conv2D(interim_d, (1, 1), activation='relu')(representations_input)
    conv2 = Conv2D(d, (1, 1))(conv1)

    previous_transformer = conv2
    for i in range(n_interaction):
        current_transformer = transformer_layers[i](previous_transformer)
        previous_transformer = current_transformer

    big_layer = BigLayer(temp, allocation_temp, n_bidders, menu_size, batch_size)
    output = big_layer([previous_transformer, bids_input])

    model = Model(inputs=[representations_input, bids_input], outputs=output)
    model.compile(loss=negative_payment_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_fn),
                  metrics=[payment_metric])

    for epoch in epochs:
        batch_data, bids = generate_values(samples=samples, n_bidders=n_bidders, n_items=n_items, dx=dx, dy=dy)

        model.fit(x=[batch_data, bids], y=tf.zeros(samples), batch_size=batch_size, verbose=0,
                  callbacks=[TqdmCallback(verbose=0)])

    batch_data, bids = generate_values(samples=samples, n_bidders=n_bidders, n_items=n_items, dx=dx, dy=dy)
    loss = model.evaluate([batch_data, bids], y=tf.zeros(samples), batch_size=batch_size)
    print(loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--m_items', type=int, default=5)
    parser.add_argument('--menu_size', type=int, default=128)

    parser.add_argument('--alloc_softmax_temperature', type=int, default=10)

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_steps', type=int, default=2000)

    return parser.parse_known_args()


if __name__ == '__main__':
    main()
