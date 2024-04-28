package org.example

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Stack
import org.tensorflow.op.core.Unstack

class TransformerBASEDLayer(
    private val numberOfHeads: Int, // n_head in the code
    private val dModel: Int, // d_in in the code
    private val feedForwardDimension: Int, // d_hidden in the code
    private val dOut: Int, // d_out in the code
    private val numberOfBidders: Long,
    private val numberOfItems: Long,
    name: String = "",
    override val hasActivation: Boolean = false
) : Layer(name) {

    private val rowTransformer by lazy {
        TransformerEncoderLayer(
            "Bidder Transformer",
            this.hasActivation,
            this.numberOfHeads,
            this.dModel,
            this.feedForwardDimension,
            dropoutRate = 0.0f
        )
    }
    private val columnTransformer by lazy {
        TransformerEncoderLayer(
            "Item Transformer",
            this.hasActivation,
            this.numberOfHeads,
            this.dModel,
            this.feedForwardDimension,
            dropoutRate = 0.0f
        )
    }


    private val conv2D =
        Conv2D(filters = feedForwardDimension, kernelSize = intArrayOf(1, 1), activation = Activations.Relu)

    private val conv2D1 = Conv2D(filters = dOut, kernelSize = intArrayOf(1, 1), activation = Activations.Linear)

    override fun build(
        tf: Ops, input: Operand<Float>, isTraining: Operand<Boolean>, numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val batchSize = if (input.asOutput().shape().size(0).toInt() == -1) this.batchSize else input.asOutput().shape().size(0)

        println(input.asOutput().shape())
        val batches = tf.unstack(input, batchSize, Unstack.axis(0L)).map {
            val rows = tf.unstack(it, numberOfBidders + 1, Unstack.axis(0L)) // (numItems) x (dOut)
                .map { row -> rowTransformer.build(tf, row, isTraining, numberOfLosses) }

            val columns = tf.unstack(it, numberOfItems, Unstack.axis(2L)) // (numBids + 1) x (dOut)
                .map { column -> columnTransformer.build(tf, column, isTraining, numberOfLosses) }

            val globalMean = tf.math.mean(input, tf.constant(intArrayOf(0, 1)))

            tf.stack((1..numberOfBidders + 1).map { row ->
                tf.stack((1..numberOfItems).map { column ->
                    tf.concat(listOf(rows[row.toInt()], columns[column.toInt()], globalMean), tf.constant(-1L))
                }, Stack.axis(0L))
            }, Stack.axis(0L)) // (numbids+1) x (numItems) x (concat)
        } // (numbatches) x (numbids+1) x (numItems) x (concat)

        val stackedBatches = tf.stack(batches, Stack.axis(0L))
        val interim = conv2D.build(tf, stackedBatches, isTraining, numberOfLosses)
        return conv2D1.build(tf, interim, isTraining, numberOfLosses)
    }
}