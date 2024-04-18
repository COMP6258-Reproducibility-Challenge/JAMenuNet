package org.example

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.ReductionType
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import org.tensorflow.Operand
import org.tensorflow.op.Ops

private const val numberOfBidders = 10L
private const val numberOfItems = 10L
private const val dx = 5L
private const val dy = 5L
private const val interimD = 7
private const val d = 6
private const val numberOfInteractionModules = 7

private const val dInp = 9L
private const val dHidden = 6L
private const val numberOfHeads = 3L
private const val dOut = 5L

private const val temp = 5.0f
private const val menuSize = 5

val input1 = Input(
    numberOfBidders + 1,
    numberOfItems,
    dx + dy
)

val conv1 = Conv2D(
    filters = interimD,
    kernelSize = intArrayOf(1, 1),
    activation = Activations.Relu
)(input1)

val conv2 = Conv2D(
    filters = d,
    kernelSize = intArrayOf(1, 1),
    activation = Activations.Linear
)(conv1)

val transformLayer = arrayOf(dInp)
    .asSequence()
    .plus(Array(numberOfInteractionModules - 1) { dHidden })
    .plus(arrayOf(dOut))
    .toList()
    .zipWithNext()
    .map {
        TransformerBASEDLayer(
            numberOfHeads = numberOfHeads.toInt(),
            dModel = it.first.toInt(),
            feedForwardDimension = dHidden.toInt(),
            dOut = it.second.toInt(),
            numberOfBidders = numberOfBidders,
            numberOfItems = numberOfItems
        )
    }
    .fold(conv2) { acc, layer ->
        layer(acc)
    }


val outputs = BigLayer(softmaxTemp = temp, numberOfItems = numberOfItems, numberOfBidders = numberOfBidders, menuSize = menuSize)(transformLayer, Input(numberOfBidders, numberOfItems))

val model = Functional.fromOutput(outputs)

private class negativePaymentLoss(reductionType: ReductionType) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>, // (nBidders) x (batchSize)
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.math.neg(tf.math.mean(tf.sum(yPred, tf.constant(0L)), tf.constant(0L)))
    }

}

fun main() {
    mnist()

    model.use {
        //TODO : Change this
        it.copile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = negativePaymentLoss(ReductionType.SUM_OVER_BATCH_SIZE),
            metric = listOf<Metric>()
        )

        it.logSummary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val loss = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).lossValue

        println("Loss: $loss")
    }
}