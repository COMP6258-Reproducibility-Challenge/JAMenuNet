package org.example

import org.jetbrains.kotlinx.dl.dataset.DataBatch
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.dataset.Dataset
import java.nio.FloatBuffer
import kotlin.math.truncate
import kotlin.random.Random

class AuctionDataset<D> internal constructor(
    private val x: Array<D>,
    private val y: FloatArray,
    private val dataLoader: DataLoader<D>,
) : Dataset() {

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copySourcesToBatch(src: Array<D>, start: Int, length: Int): Array<FloatArray> {
        return Array(length) { index -> dataLoader.load(src[start + index]).first }
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyLabelsToBatch(src: FloatArray, start: Int, length: Int): FloatArray {
        return FloatArray(length) { src[start + it] }
    }

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    override fun split(splitRatio: Double): Pair<AuctionDataset<D>, AuctionDataset<D>> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        val train = AuctionDataset(
            x.copyOfRange(0, trainDatasetLastIndex),
            y.copyOfRange(0, trainDatasetLastIndex),
            dataLoader
        )
        val test = AuctionDataset(
            x.copyOfRange(trainDatasetLastIndex, x.size),
            y.copyOfRange(trainDatasetLastIndex, y.size),
            dataLoader
        )

        return Pair(train, test)
    }

    /** Returns amount of data rows. */
    override fun xSize(): Int {
        return x.size
    }

    /** Returns row by index [idx]. */
    override fun getX(idx: Int): FloatArray {
        return dataLoader.load(x[idx]).first
    }

    /** Returns label as [FloatArray] by index [idx]. */
    override fun getY(idx: Int): Float {
        return y[idx]
    }

    override fun shuffle(): AuctionDataset<D> {
        x.shuffle(Random(12L))
        y.shuffle(Random(12L))
        return this
    }

    override fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch {
        return DataBatch(
            copySourcesToBatch(x, batchStart, batchLength),
            copyLabelsToBatch(y, batchStart, batchLength),
            batchLength
        )
    }

    fun create(
        features: Array<D>,
        labels: FloatArray,
        dataLoader: DataLoader<D>
    ): AuctionDataset<D> {
        check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

        return AuctionDataset(features, labels, dataLoader)
    }
}