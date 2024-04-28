package org.example

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.initializer.Ones
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.KVariable
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.layer.ParametrizedLayer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.Softmax
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.Mean
import javax.naming.OperationNotSupportedException

// Implemented using https://medium.com/@prudhviraju.srivatsavaya/implementing-multiheaded-attention-a321dcb5aab8
class TransformerEncoderLayer(name: String,
                              override val hasActivation: Boolean,
                              private val numberOfHeads: Int,
                              private val dModel: Int,
                              private val feedForwardDimension: Int,
                              private val dropoutRate: Float = 0.1f) : Layer(name) {

    private val mha by lazy { MultiHeadAttention(this.hasActivation, this.name, this.numberOfHeads, this.dModel) }
    private val feedForward1 by lazy { Dense(feedForwardDimension) }
    private val feedForward2 by lazy { Dense(dModel, activation = Activations.Linear) }
    private val layerNorm1 by lazy { LayerNormalization(name = this.name, hasActivation = this.hasActivation, epsilon = 0.000001) }
    private val layerNorm2 by lazy { LayerNormalization(name = this.name, hasActivation = this.hasActivation, epsilon = 0.000001) }
    private val dropout1 by lazy { Dropout(dropoutRate) }
    private val dropout2 by lazy { Dropout(dropoutRate) }

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // input shape = [10, 6]
        var attentionOutput = this.mha.build(tf, listOf(input, input, input), isTraining, numberOfLosses)
        // [10, 9]
        attentionOutput = this.dropout1.build(tf, attentionOutput, isTraining, numberOfLosses)
        // [10, 9]
        val out1 = this.layerNorm1.build(tf, tf.math.add(input, attentionOutput), isTraining, numberOfLosses)

        var feedForwardOutput = this.feedForward2.build(tf, feedForward1.build(tf, out1, isTraining, numberOfLosses), isTraining, numberOfLosses)
        feedForwardOutput = this.dropout2.build(tf, feedForwardOutput, isTraining, numberOfLosses)
        val out2 = this.layerNorm2.build(tf, tf.math.add(out1, feedForwardOutput), isTraining, numberOfLosses)

        return out2
    }

    // This follows the Keras implementation https://github.com/keras-team/keras/blob/v3.2.0/keras/layers/normalization/layer_normalization.py
    // axis is always -1
    private class LayerNormalization(name: String, override val hasActivation: Boolean,
                                     val center: Boolean = true,
                                     val epsilon: Double = 0.001,
                                     val scale: Boolean = true,
                                     val gammaInitializer: Initializer = Ones(),
                                     val betaInitializer: Initializer = Zeros(),
                                     val gammaRegularizer: Regularizer? = null,
                                     val betaRegularizer: Regularizer? = null)
        : Layer(name), NoGradients, ParametrizedLayer {
        var gamma: KVariable? = null
        var beta: KVariable? = null

        override val variables: List<KVariable>
            get() = listOfNotNull(gamma, beta)

        override fun build(
            tf: Ops,
            input: Operand<Float>,
            isTraining: Operand<Boolean>,
            numberOfLosses: Operand<Float>?
        ): Operand<Float> {
            val inputShape = input.asOutput().shape()
            val weightShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1))

            val fanIn = Int.MIN_VALUE
            val fanOut = Int.MIN_VALUE

            if (scale) {
                gamma = createVariable(
                    tf,
                    "gamma",
                    weightShape,
                    fanIn,
                    fanOut,
                    gammaInitializer,
                    gammaRegularizer
                )
            }

            if (center) {
                beta = createVariable(
                    tf,
                    "beta",
                    weightShape,
                    fanIn,
                    fanOut,
                    betaInitializer,
                    betaRegularizer
                )
            }

            val mean = tf.math.mean(input, tf.constant(-1), Mean.keepDims(true))
            val variance = tf.math.mean(tf.math.square(tf.math.sub(input, mean)), tf.constant(-1), Mean.keepDims(true))
            var invStd: Operand<Float> = tf.math.rsqrt(tf.math.add(variance, tf.constant(epsilon.toFloat())))

            gamma?.let {
                invStd = tf.math.mul(invStd, it.variable)
            }

            var res: Operand<Float> = tf.math.mul(tf.math.neg(mean), invStd)

            beta?.let {
                res = tf.math.add(res, it.variable)
            }

            return tf.math.add(tf.math.mul(input, invStd), res)
        }


        fun createVariable(
            tf: Ops,
            variableName: String,
            shape: Shape,
            fanIn: Int,
            fanOut: Int,
            initializer: Initializer,
            regularizer: Regularizer?
        ): KVariable {
            val tfVariable = tf.withName(variableName).variable(shape, Float::class.javaObjectType)
            val initializerOperation = initializer.apply(fanIn, fanOut, tf, tfVariable, variableName)
            return KVariable(
                name = variableName,
                shape = shape,
                variable = tfVariable,
                initializerOperation = initializerOperation,
                regularizer = regularizer
            )
        }
    }

    private class MultiHeadAttention(override val hasActivation: Boolean,
                                     name: String,
                                     val numberOfHeads: Int,
                                     val dModel: Int) : Layer(name) {

        val depth by lazy { dModel / numberOfHeads }
        val queryDense by lazy { Dense(outputSize = dModel, activation = Activations.Linear) }
        val keyDense by lazy { Dense(outputSize = dModel, activation = Activations.Linear) }
        val valueDense by lazy { Dense(outputSize = dModel, activation = Activations.Linear) }
        val dense by lazy { Dense(outputSize = dModel, activation = Activations.Linear) }

        override fun build(
            tf: Ops,
            input: Operand<Float>,
            isTraining: Operand<Boolean>,
            numberOfLosses: Operand<Float>?
        ): Operand<Float> {
           throw OperationNotSupportedException("Use the build with list input")
        }

        fun splitHeads(tf: Ops, input: Operand<Float>, batchSize: Long): Operand<Float> {
            val reshapedInput = tf.reshape(input, tf.constant(longArrayOf(batchSize, -1, this.numberOfHeads.toLong(), this.depth.toLong())))
            return tf.linalg.transpose(reshapedInput, tf.constant(intArrayOf(0, 2, 1, 3)))
        }

        // We don't need the mask
        fun scaledDotProductAttention(tf: Ops,
                                      query: Operand<Float>,
                                      key: Operand<Float>,
                                      value: Operand<Float>): Pair<Operand<Float>, Operand<Float>> {
            val matMulQueryKey = tf.linalg.matMul(query, key, MatMul.transposeB(true))
            val dk = key.asOutput().shape().size(key.asOutput().shape().numDimensions() - 1).toFloat()
            val scaledAttentionLogits: Operand<Float> = tf.math.div(matMulQueryKey, tf.constant(dk))

            val attentionWeights = Softmax().forward(tf, scaledAttentionLogits)
            val output = tf.linalg.matMul(attentionWeights, value)

            return Pair(output, attentionWeights)
        }

        override fun build(
            tf: Ops,
            input: List<Operand<Float>>,
            isTraining: Operand<Boolean>,
            numberOfLosses: Operand<Float>?
        ): Operand<Float> {
            // Don't use mask
            assert(input.size == 3)
            var query = input[0]
            var key = input[1]
            var value = input[2]

            val batchSize = query.asOutput().shape().size(0)

            query = this.queryDense.build(tf, query, isTraining, numberOfLosses)
            key = this.keyDense.build(tf, key, isTraining, numberOfLosses)
            value = this.valueDense.build(tf, value, isTraining, numberOfLosses)

            query = splitHeads(tf, query, batchSize)
            key = splitHeads(tf, key, batchSize)
            value = splitHeads(tf, value, batchSize)

            val (scaledAttention, _) = scaledDotProductAttention(tf, query, key, value)
            val transposedAttention = tf.linalg.transpose(scaledAttention, tf.constant(intArrayOf(0, 2, 1, 3)))
            val concatAttention = tf.reshape(transposedAttention, tf.constant(longArrayOf(batchSize, -1, this.dModel.toLong())))
            val output = this.dense.build(tf, concatAttention, isTraining, numberOfLosses)

            return output // We seem not to need the attentionWeights
        }
    }
}