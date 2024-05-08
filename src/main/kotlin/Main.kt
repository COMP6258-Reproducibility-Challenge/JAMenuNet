package org.example

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.ReductionType
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import kotlin.math.exp
import kotlin.random.Random

val TEST_BATCH_SIZE = 50L
val TRAINING_BATCH_SIZE = 50L
val EPOCHS: Int = 50
private const val numberOfBidders = 10L
private const val numberOfItems = 10L
private const val dx = 5L
private const val dy = dx
private const val interimD = 7
private const val d = 6
private const val numberOfInteractionModules = 2

private const val dHidden = 5L
private const val numberOfHeads = 1L
private const val menuSize = 1
private const val dOut = 2L * menuSize.toLong() + 1L

private const val temp = 5.0f

private val transformerBASEDLayers = arrayOf(d)
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
            numberOfItems = numberOfItems,
            batchSize = TRAINING_BATCH_SIZE
        )
    }

val input = Input(
    numberOfBidders + 1,
    numberOfItems,
    dx + dy + 1
)

val representationsLayer = GetRepresentationsLayer(dx = dx, dy = dy)(input)
val bidsLayer = GetBidsLayer(dx = dx, dy = dy)(input)

val conv1 = Conv2D(
    filters = interimD,
    kernelSize = intArrayOf(1, 1),
    activation = Activations.Relu
)(representationsLayer)

val conv2 = Conv2D(
    filters = d,
    kernelSize = intArrayOf(1, 1),
    activation = Activations.Linear
)(conv1)

val transformLayer = transformerBASEDLayers
    .fold(conv2) { acc, layer ->
        layer(acc)
    }

private val layer = BigLayer(
    softmaxTemp = temp,
    numberOfItems = numberOfItems,
    numberOfBidders = numberOfBidders,
    menuSize = menuSize,
    dx = dx,
    dy = dy,
    interimD = interimD,
    d = d,
    numberOfInteractionModules = numberOfInteractionModules,
    dHidden = dHidden,
    numberOfHeads = numberOfHeads,
    dOut = dOut,
    batchSize = TRAINING_BATCH_SIZE
)

val outputs =
    layer(transformLayer, bidsLayer)

val model = Functional.fromOutput(outputs)

private class NegativePaymentLoss(reductionType: ReductionType) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>, // (nBidders) x (batchSize)
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.math.neg(tf.math.mean(tf.sum(yPred, tf.constant(0L)), tf.constant(0L)))
    }
}

private class PaymentMetric(reductionType: ReductionType) : Metric(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>, // (nBidders) x (batchSize)
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        return tf.math.mean(tf.sum(yPred, tf.constant(0L)), tf.constant(0L))
    }
}

private fun sigmoid(x: Float): Float {
    return 1f / (1f + exp(-x))
}

fun main() {

    // X, Y, bids
    // X - samples x bidders x dx
    // Y - samples x items x dy
    // bids - samples x bidders x items

    val samples = 100
    val X = (1..samples).map {
        (1..numberOfBidders).map {
            (1..dx).map { Random.nextFloat() * 2f - 1f }
        }.plusElement((1..dx).map { 1f })
    }
    val Y = (1..samples).map {
        (1..numberOfItems).map {
            (1..dy).map { Random.nextFloat() * 2 - 1 }
        }
    }
    val bids = (0..<samples).map { samp ->
        (0..<numberOfBidders.toInt()).map { bidder ->
            (0..<numberOfItems.toInt()).map { item ->
                Random.nextFloat() * sigmoid((0..<dx.toInt()).map {
                    X[samp][bidder][it] * // (dx)
                            Y[samp][item][it] // (dy)
                }.sum())
            }
        }.plusElement((1..numberOfItems).map { 1f })
    }


    val x = (0..<samples).map { samp ->
        (0..<numberOfBidders.toInt() + 1).map { bidder ->
            (0..<numberOfItems.toInt()).map { item ->
                X[samp][bidder].toFloatArray()
                    .plus(Y[samp][item].toFloatArray())
                    .plus(bids[samp][bidder][item])
            }.toTypedArray()
        }.toTypedArray()
    }.toTypedArray()

    val y = x.map { 0f }.toFloatArray()
    val dataset = AuctionDataset(x, y, AuctionDataLoader())
    val (train, test) = dataset.split(0.8)
    model.use {
        //TODO : Change this
        it.compile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = NegativePaymentLoss(ReductionType.SUM_OVER_BATCH_SIZE),
            metric = PaymentMetric(ReductionType.SUM_OVER_BATCH_SIZE)
        )

        it.logSummary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE.toInt())

        println("Start training")
        layer.isTraining(false)

        val loss = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE.toInt()).lossValue

        println("Loss: $loss")
    }
}