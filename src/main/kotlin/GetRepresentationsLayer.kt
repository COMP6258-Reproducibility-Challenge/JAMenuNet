package org.example

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

class GetRepresentationsLayer(name: String = "",
                              private val dx: Long,
                              private val dy: Long,
                              override val hasActivation: Boolean = false): Layer(name) {

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val (representations, _) = tf.splitV(
            input,
            tf.constant(intArrayOf((dx + dy).toInt(), 1)),
            tf.constant(3),
            2L
        ).toList()

        return representations
    }
}