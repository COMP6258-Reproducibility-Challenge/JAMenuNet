package org.example

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.Softmax
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gather
import org.tensorflow.op.core.Squeeze
import org.tensorflow.op.core.Stack
import org.tensorflow.op.core.Unstack

class BigLayer(
    name: String = "",
    private val softmaxTemp: Float,
    private val numberOfBidders: Long,
    private val numberOfItems: Long,
    private val menuSize: Int,
    override val hasActivation: Boolean = false

) : Layer(name) {

    override fun build(
        tf: Ops,
        input: List<Operand<Float>>, // inputs[0] should be in format (nBatches) x (nBidders + 1) x (nItems) x (2 * menuSize+1)
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val training = isTraining.asOutput().tensor().booleanValue()

        var (menu, weights, boosts) = tf.splitV(
            input[0],
            tf.constant(intArrayOf(menuSize, 1, menuSize)),
            tf.constant(3),
            3L
        ).toList()
        menu = tf.stack(tf.unstack(menu, numberOfItems, Unstack.axis(2L)).map { itemInfo ->
            tf.stack(tf.unstack(itemInfo, menuSize.toLong(), Unstack.axis(2L)).map {
                tf.nn.softmax(tf.math.mul(menu, tf.constant(softmaxTemp)))
            }, Stack.axis(2L))
        }, Stack.axis(2L))
        menu = tf.linalg.transpose(
            tf.slice(menu, tf.constant(intArrayOf(0, 0, 0, 0)), tf.constant(intArrayOf(-1, numberOfBidders.toInt(), -1, -1))),
            tf.constant(intArrayOf(0, 3, 1, 2))) // (batchSize) x (menuSize) x (nBidders) x (nItems)

        if (weights.asOutput().shape().numDimensions() == 4) weights = tf.squeeze(weights, Squeeze.axis(listOf(3)))
        weights = tf.math.sigmoid(tf.math.mean(weights, tf.constant(2L)))
        weights = tf.slice(weights, tf.constant(intArrayOf(0, 0)), tf.constant(intArrayOf(-1, numberOfBidders.toInt()))) // (batchSize) x (nBidders)

        boosts = tf.math.mean(boosts, tf.constant(2L))
        boosts = tf.math.mean(boosts, tf.constant(1L))
        boosts = Dense(outputSize = menuSize, activation = Activations.Relu).build(tf, boosts, isTraining, numberOfLosses)
        boosts = Dense(outputSize = menuSize, activation = Activations.Linear).build(tf, boosts, isTraining, numberOfLosses) // (batchSize) x (menuSize)

        var bids = input[1] // (batchSize) x (nBidders) x (nItems)
        bids = tf.expandDims(bids, tf.constant(1L)) // (batchSize) x 1 x (nBidders) x (nItems)

        return if (training) trainingCalc(tf, menu, weights, boosts, bids)
        else testCalc(tf, menu, weights, boosts, bids)
    }

    private fun trainingCalc(tf: Ops, menu: Operand<Float>, weights: Operand<Float>, boosts: Operand<Float>, bids: Operand<Float>): Operand<Float> {

        val utils = tf.sum(tf.math.mul(menu, bids), tf.constant(2L)) // (batchSize) x (menuSize) x (nBidders)
        val welfarePerBidder = tf.math.mul(tf.expandDims(weights, tf.constant(1L)), utils) // (batchSize) x (menuSize) x (nBidders)
        val welfare = tf.sum(welfarePerBidder, tf.constant(2L)) // (batchSize) x (menuSize)
        val boostedWelfare = tf.math.add(welfare, boosts) // (batchSize) x (menuSize)
        val chosenAllocation = tf.nn.softmax(tf.math.mul(boostedWelfare, tf.constant(softmaxTemp))) // (batchSize) x (menuSize)
        val expectedWelfarePerBidder = tf.sum(tf.math.mul(welfarePerBidder, tf.expandDims(chosenAllocation, tf.constant(-1L))), tf.constant(1L)) // (batchSize) x (nBidders)

        val batchSize = menu.asOutput().shape().size(0)

        var mask: Operand<Float> = tf.oneHot(
            tf.constant((0..<numberOfBidders).toList().toTypedArray().toLongArray()),
            tf.constant(numberOfBidders.toInt()),
            tf.constant(0f),
            tf.constant(1f)) // (nBidders) x (nBidders)
        mask = tf.stack(tf.unstack(mask, numberOfBidders, Unstack.axis(0L)).map { tensor ->
            tf.stack((1..batchSize).map { tensor }, Stack.axis(0L))
        }, Stack.axis(0L)) // (nBidders) x (batchSize) x (nBidders)

        val repeatBoosts = tf.tile(tf.expandDims(boosts, tf.constant(0L)), tf.constant(longArrayOf(numberOfBidders, 1L, 1L))) // (nBidders) x (batchSize) x (menuSize)

        val maskedWelfarePerBidder = tf.math.mul(tf.tile(tf.expandDims(expectedWelfarePerBidder, tf.constant(0L)),
            tf.constant(longArrayOf(numberOfBidders, 1L, 1L))), mask) // (nBidders) x (batchSize) x (nBidders)
        val maskedWelfare = tf.math.mul(tf.tile(tf.expandDims(welfarePerBidder, tf.constant(0L)),
            tf.constant(longArrayOf(numberOfBidders, 1L, 1L, 1L))),
            tf.expandDims(mask, tf.constant(2L))) // (nBidders) x (batchSize) x (menuSize) x (nBidders)
        val totalMaskedWelfare = tf.sum(maskedWelfare, tf.constant(-1L)) // (nBidders) x (batchSize) x (menuSize)
        val maskedChosenAllocation = Softmax().forward(tf, tf.math.mul(tf.math.add(totalMaskedWelfare, repeatBoosts),
        tf.constant(softmaxTemp))) // (nBidders) x (batchSize) x (menuSize)

        val maskedExpectedWelfarePerBidder = tf.sum(tf.math.mul(maskedWelfare,
            tf.expandDims(maskedChosenAllocation, tf.constant(-1))), tf.constant(2L)) // (nBidders) x (batchSize) x (nBidders)

        val sum1 = tf.math.add(tf.sum(maskedExpectedWelfarePerBidder, tf.constant(-1L)),
            tf.sum(tf.math.mul(maskedChosenAllocation, repeatBoosts), tf.constant(-1L))) // (nBidders) x (batchSize)
        val sum2 = tf.math.add(tf.sum(maskedWelfarePerBidder, tf.constant(-1L)),
            tf.tile(tf.expandDims(tf.sum(tf.math.mul(chosenAllocation, boosts), tf.constant(1L)), tf.constant(0L)),
                tf.constant(longArrayOf(numberOfBidders, 1L)))) // (nBidders) x (batchSize)
        val payments = tf.math.mul(tf.math.div(tf.constant(1f), tf.linalg.transpose(weights, tf.constant(longArrayOf(1L, 0L)))),
            tf.math.sub(sum1, sum2)) // (nBidders) x (batchSize)

        // TODO: We might also want to return the allocations etc.
        return payments
    }

    private fun testCalc(tf: Ops, menu: Operand<Float>, weights: Operand<Float>, boosts: Operand<Float>, bids: Operand<Float>): Operand<Float> {

        val utils = tf.sum(tf.math.mul(menu, bids), tf.constant(2L)) // (batchSize) x (menuSize) x (nBidders)
        val welfarePerBidder = tf.math.mul(tf.expandDims(weights, tf.constant(1L)), utils) // (batchSize) x (menuSize) x (nBidders)
        val welfare = tf.sum(welfarePerBidder, tf.constant(2L)) // (batchSize) x (menuSize)
        val boostedWelfare = tf.math.add(welfare, boosts) // (batchSize) x (menuSize)

        val chosenAllocationIndex = tf.math.argMax(boostedWelfare, tf.constant(1L)) // (batchSize)
//        val chosenAllocation = tf.gather(menu, argMaxIndex, tf.constant(1L))
        val expectedWelfarePerBidder = tf.gather(welfarePerBidder, chosenAllocationIndex, tf.constant(1L), Gather.batchDims(1L)) // (batchSize) x (nBidders)

        val batchSize = menu.asOutput().shape().size(0)

        var mask: Operand<Float> = tf.oneHot(
            tf.constant((0..<numberOfBidders).toList().toTypedArray().toLongArray()),
            tf.constant(numberOfBidders.toInt()),
            tf.constant(0f),
            tf.constant(1f)) // (nBidders) x (nBidders)
        mask = tf.stack(tf.unstack(mask, numberOfBidders, Unstack.axis(0L)).map { tensor ->
            tf.stack((1..batchSize).map { tensor }, Stack.axis(0L))
        }, Stack.axis(0L)) // (nBidders) x (batchSize) x (nBidders)

        val repeatBoosts = tf.tile(tf.expandDims(boosts, tf.constant(0L)), tf.constant(longArrayOf(numberOfBidders, 1L, 1L))) // (nBidders) x (batchSize) x (menuSize)

        val maskedWelfarePerBidder = tf.math.mul(tf.tile(tf.expandDims(expectedWelfarePerBidder, tf.constant(0L)),
            tf.constant(longArrayOf(numberOfBidders, 1L, 1L))), mask) // (nBidders) x (batchSize) x (nBidders)
        val maskedWelfare = tf.math.mul(tf.tile(tf.expandDims(welfarePerBidder, tf.constant(0L)),
            tf.constant(longArrayOf(numberOfBidders, 1L, 1L, 1L))),
            tf.expandDims(mask, tf.constant(2L))) // (nBidders) x (batchSize) x (menuSize) x (nBidders)
        val totalMaskedWelfare = tf.sum(maskedWelfare, tf.constant(-1L)) // (nBidders) x (batchSize) x (menuSize)
        val maskedChosenAllocationIndices = tf.math.argMax(tf.math.mul(tf.math.add(totalMaskedWelfare, repeatBoosts),
            tf.constant(softmaxTemp)), tf.constant(-1L)) // (nBidders) x (batchSize)
        val maskedExpectedWelfare = tf.gather(totalMaskedWelfare, maskedChosenAllocationIndices, tf.constant(2L), Gather.batchDims(2L)) // (nBidders) x (batchSize)
        val maskedBoosts = tf.gather(repeatBoosts, maskedChosenAllocationIndices, tf.constant(2L), Gather.batchDims(2L)) // (nBidders) x (batchSize)

        val regularBoosts = tf.gather(repeatBoosts, chosenAllocationIndex, tf.constant(2L), Gather.batchDims(2L)) // (nBidders) x (batchSize)

        val sum1 = tf.math.add(maskedExpectedWelfare, maskedBoosts) // (nBidders) x (batchSize)
        val sum2 = tf.math.add(tf.sum(maskedWelfarePerBidder, tf.constant(2L)), regularBoosts) // (nBidders) x (batchSize)
        val payments = tf.math.mul(tf.math.div(tf.constant(1f), tf.linalg.transpose(weights, tf.constant(longArrayOf(1L, 0L)))),
            tf.math.sub(sum1, sum2)) // (nBidders) x (batchSize)

        return payments
    }

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        throw UnsupportedOperationException("BigLayer requires list of inputs")
    }
}