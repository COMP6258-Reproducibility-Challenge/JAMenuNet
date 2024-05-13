import keras
from keras import Input, Model
from keras.src.layers import Conv2D
import tensorflow as tf
from tqdm.keras import TqdmCallback

from big_layer import BigLayer
from transformer_based_layer import TransformerBasedLayer


def negative_payment_loss(y_true, y_pred):
    return tf.negative(tf.reduce_mean(tf.reduce_sum(y_pred, axis=0), axis=0))


def payment_metric(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(y_pred, axis=0), axis=0)


def generate_values(samples, n_bidders, n_items, dx, dy):
    x = tf.concat([tf.random.uniform((samples, n_bidders, dx), minval=-1.0, maxval=1.0),
                   tf.tile(tf.expand_dims(tf.expand_dims(tf.repeat(1.0, dx), axis=0), axis=0), [samples, 1, 1])]
                  , axis=1)
    y = tf.random.uniform((samples, n_items, dy), minval=-1.0, maxval=1.0)
    upper = []
    for s in range(samples):
        bidders = []
        for b in range(n_bidders):
            items = []
            for i in range(n_items):
                items.append(tf.sigmoid(tf.reduce_sum(x[s][b] * y[s][i], axis=-1)))
            items = tf.stack(items)
            bidders.append(items)
        bidders = tf.stack(bidders)
        upper.append(bidders)
    upper = tf.stack(upper)
    bids = tf.concat([tf.multiply(tf.random.uniform((samples, n_bidders, n_items)), upper),
                      tf.tile(tf.expand_dims(tf.expand_dims(tf.repeat(1.0, n_items), axis=0), axis=0), [samples, 1, 1])]
                     , axis=1)

    batch_data = []
    for i in range(samples):
        bidder_data = []
        for j in range(n_bidders + 1):
            item_data = []
            for k in range(n_items):
                item_data.append(tf.concat([x[i][j], y[i][k]], axis=0))
            item_data = tf.stack(item_data)
            bidder_data.append(item_data)
        bidder_data = tf.stack(bidder_data)
        batch_data.append(bidder_data)
    batch_data = tf.stack(batch_data)

    return batch_data, bids


def main():
    batch_size = 50 #2048  # Make sure this is a divisor of samples
    epochs = 2  # train_steps
    samples = 100 #32768  # train_sample_num = eval_sample_num
    seed = 3

    start_lr = 1e-8  # This is static in the code
    end_lr = 3e-4  # lr
    lr_fn = keras.optimizers.schedules.PolynomialDecay(start_lr, end_lr, 100, power=1.0)

    n_bidders = 2
    n_items = 5  # m_items
    dx = 10
    dy = dx
    d_hidden = 64
    d = d_hidden
    interim_d = d
    n_interaction = 3  # n_layer

    n_heads = 4  # n_head
    menu_size = 128  # menu_size
    d_out = 2 * menu_size + 1

    temp = 500.0  # init_softmax_temperature
    allocation_temp = 10.0  # alloc_softmax_temperature

    keras.utils.set_random_seed(seed)

    transformer_layers = []
    for i in range(n_interaction):
        if i == 0:
            transformer_layers.append(TransformerBasedLayer(n_heads, d, d_hidden, d_hidden, n_bidders, n_items, batch_size))
        elif i == n_interaction - 1:
            transformer_layers.append(TransformerBasedLayer(n_heads, d_hidden, d_hidden, d_out, n_bidders, n_items, batch_size))
        else:
            transformer_layers.append(TransformerBasedLayer(n_heads, d_hidden, d_hidden, d_hidden, n_bidders, n_items, batch_size))

    representations_input = Input(shape=(n_bidders + 1, n_items, dx + dy), batch_size=batch_size)
    bids_input = Input(shape=(n_bidders + 1, n_items), batch_size=batch_size)

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
                  optimizer=keras.optimizers.Adam(learning_rate=start_lr),
                  metrics=[payment_metric])

    batch_data, bids = generate_values(samples=samples, n_bidders=n_bidders, n_items=n_items, dx=dx, dy=dy)

    for epoch in range(epochs):
        model.fit(x=[batch_data, bids], y=tf.zeros(samples), batch_size=batch_size, verbose=0, callbacks=[TqdmCallback(verbose=0)])

    batch_data, bids = generate_values(samples=samples, n_bidders=n_bidders, n_items=n_items, dx=dx, dy=dy)
    loss = model.evaluate([batch_data, bids], y=tf.zeros(samples), batch_size=batch_size)
    print(loss)


if __name__ == '__main__':
    main()
