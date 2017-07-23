package io.anserini.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Conv1d {

    private int inChannels;
    private int outChannels;
    private int kernelSize;
    private int stride;
    private int padding;

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    public INDArray forward(INDArray input, INDArray filter) {
        // TODO Sanity checking the input
        // TODO add support for non-one stride

        if (padding > 0) {
            INDArray paddedArray = Nd4j.zeros(input.rows(), input.columns() + 2*padding);
            INDArrayIndex index[] = {NDArrayIndex.all(), NDArrayIndex.interval(padding, padding + input.columns())};
            paddedArray.put(index, input);
            input = paddedArray;
        }

        int outputDim = input.shape()[1] - kernelSize + 1;
        INDArray result = Nd4j.zeros(outputDim);

        for (int i = 0; i < outputDim; i++) {
            INDArrayIndex interval = NDArrayIndex.interval(i, i + kernelSize);
            INDArray intervalElements = input.get(NDArrayIndex.all(), interval);
            INDArray elementWiseProduct = intervalElements.mul(filter);
            double sum = elementWiseProduct.sumNumber().doubleValue();
            result.putScalar(i, sum);
        }

        return result;
    }
}

