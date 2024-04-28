package org.example

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.impl.util.flattenFloats

class AuctionDataLoader : DataLoader<Array<Array<FloatArray>>> {

    override fun load(dataSource: Array<Array<FloatArray>>): Pair<FloatArray, TensorShape> {
        val shape = TensorShape(longArrayOf(dataSource.size.toLong(),
            dataSource[0].size.toLong(),
            dataSource[0][0].size.toLong()))
        return dataSource.flattenFloats() to shape
    }
}