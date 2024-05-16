import keras
import tensorflow as tf


class BigLayer(keras.layers.Layer):
    def __init__(self, softmax_temp, allocation_softmax_temp, n_bidders, menu_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.menu_size = menu_size
        self.softmax_temp = softmax_temp
        self.allocation_softmax_temp = allocation_softmax_temp
        self.n_bidders = n_bidders
        self.dense1 = keras.layers.Dense(menu_size, activation=keras.activations.relu)
        self.dense2 = keras.layers.Dense(menu_size)

    def call(self, inputs, training=None):
        representations, bids = inputs
        menu, weights, boosts = tf.split(representations, num_or_size_splits=[self.menu_size, 1, self.menu_size],
                                         axis=3)

        menu = tf.math.softmax(menu * self.allocation_softmax_temp, axis=1)
        menu = tf.transpose(menu, [0, 3, 1, 2])
        menu = menu[:, :, :-1, :]

        if weights.shape.rank == 4:
            weights = tf.squeeze(weights, axis=3)
        weights = tf.sigmoid(tf.reduce_mean(weights, axis=2))
        weights = weights[:, :-1]

        boosts = tf.reduce_mean(boosts, axis=2)
        boosts = tf.reduce_mean(boosts, axis=1)
        boosts = self.dense1(boosts)
        boosts = self.dense2(boosts)

        bids = tf.squeeze(bids)
        bids = tf.expand_dims(bids, axis=1)

        if training:
            return self.__training_calc(menu, weights, boosts, bids)
        else:
            return self.__test_calc(menu, weights, boosts, bids)

    def __training_calc(self, menu, weights, boosts, bids):
        utils = tf.reduce_sum(tf.multiply(menu, bids), axis=-1)
        welfare_per_bidder = tf.multiply(tf.expand_dims(weights, axis=1), utils)
        welfare = tf.reduce_sum(welfare_per_bidder, axis=2)
        boosted_welfare = tf.add(welfare, boosts)
        chosen_allocation = tf.math.softmax(tf.multiply(boosted_welfare, self.softmax_temp))
        expected_welfare_per_bidder = tf.reduce_sum(
            tf.multiply(welfare_per_bidder, tf.expand_dims(chosen_allocation, axis=-1)),
            axis=1)

        mask = tf.one_hot(range(self.n_bidders), self.n_bidders, 0.0, 1.0)
        mask = tf.unstack(mask, axis=0)
        repeat_mask = []
        for row in mask:
            batch_mask = []
            for _ in range(self.batch_size):
                batch_mask.append(row)
            batch_mask = tf.stack(batch_mask, axis=0)
            repeat_mask.append(batch_mask)
        mask = tf.stack(repeat_mask, axis=0)

        repeat_boosts = tf.tile(tf.expand_dims(boosts, axis=0), [self.n_bidders, 1, 1])

        masked_welfare_per_bidder = tf.multiply(
            tf.tile(tf.expand_dims(expected_welfare_per_bidder, axis=0), [self.n_bidders, 1, 1]),
            mask)
        masked_welfare = tf.multiply(
            tf.tile(tf.expand_dims(welfare_per_bidder, axis=0), [self.n_bidders, 1, 1, 1]),
            tf.expand_dims(mask, axis=2))
        total_masked_welfare = tf.reduce_sum(masked_welfare, axis=-1)
        masked_chosen_allocation = tf.math.softmax(tf.multiply(
            tf.add(total_masked_welfare, repeat_boosts),
            self.softmax_temp))
        masked_expected_welfare_per_bidder = tf.reduce_sum(tf.multiply(
            masked_welfare,
            tf.expand_dims(masked_chosen_allocation, axis=-1)), axis=2)

        sum1 = tf.add(
            tf.reduce_sum(masked_expected_welfare_per_bidder, axis=-1),
            tf.reduce_sum(tf.multiply(masked_chosen_allocation, repeat_boosts), axis=-1))
        sum2 = tf.add(
            tf.reduce_sum(masked_welfare_per_bidder, axis=-1),
            tf.tile(tf.expand_dims(
                tf.reduce_sum(tf.multiply(chosen_allocation, boosts), axis=1), axis=0), [self.n_bidders, 1]))
        payments = tf.multiply(tf.divide(1.0, tf.transpose(weights, [1, 0])), tf.subtract(sum1, sum2))

        return payments

    def __test_calc(self, menu, weights, boosts, bids):
        utils = tf.reduce_sum(tf.multiply(menu, bids), axis=-1)
        welfare_per_bidder = tf.multiply(tf.expand_dims(weights, axis=1), utils)
        welfare = tf.reduce_sum(welfare_per_bidder, axis=2)
        boosted_welfare = tf.add(welfare, boosts)
        chosen_allocation_index = tf.argmax(boosted_welfare, axis=1)
        expected_welfare_per_bidder = tf.gather(welfare_per_bidder, chosen_allocation_index, axis=1, batch_dims=1)

        mask = tf.one_hot(range(self.n_bidders), self.n_bidders, 0.0, 1.0)
        mask = tf.unstack(mask, axis=0)
        repeat_mask = []
        for row in mask:
            batch_mask = []
            for _ in range(self.batch_size):
                batch_mask.append(row)
            batch_mask = tf.stack(batch_mask, axis=0)
            repeat_mask.append(batch_mask)
        mask = tf.stack(repeat_mask, axis=0)

        repeat_boosts = tf.tile(tf.expand_dims(boosts, axis=0), [self.n_bidders, 1, 1])

        masked_welfare = tf.multiply(
            tf.tile(tf.expand_dims(welfare_per_bidder, axis=0), [self.n_bidders, 1, 1, 1]),
            tf.expand_dims(mask, axis=2))
        total_masked_welfare = tf.reduce_sum(masked_welfare, axis=-1)
        masked_chosen_allocation_indices = tf.argmax(
            tf.add(total_masked_welfare, repeat_boosts), axis=-1)
        masked_expected_welfare = tf.gather(total_masked_welfare, masked_chosen_allocation_indices,
                                            axis=2, batch_dims=2)
        masked_boosts = tf.gather(repeat_boosts, masked_chosen_allocation_indices, axis=2, batch_dims=2)

        chosen_allocation_index = tf.tile(tf.expand_dims(chosen_allocation_index, axis=0), [self.n_bidders, 1])
        regular_boosts = tf.gather(repeat_boosts, chosen_allocation_index, axis=2, batch_dims=2)

        sum1 = tf.add(masked_expected_welfare, masked_boosts)
        sum2 = []
        for i in range(self.n_bidders):
            sum2.append(tf.subtract(tf.reduce_sum(expected_welfare_per_bidder, axis=1), expected_welfare_per_bidder[:, i]))
        tf.stack(sum2, axis=0)
        sum2 = tf.add(sum2, regular_boosts)
        payments = tf.multiply(tf.divide(1.0, tf.transpose(weights, [1, 0])), tf.subtract(sum1, sum2))

        return payments
