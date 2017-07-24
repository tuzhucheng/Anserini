package io.anserini.nn;

/**
 * Created by jackzhang on 7/23/17.
 */
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

public class DL4JConv1d {
    private int inChannels;
    private int outChannels;
    private int kernelSize;
    private int stride;
    private int padding;
    private MultiLayerNetwork model;

    public DL4JConv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding, INDArray filters, INDArray biases) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new ConvolutionLayer.Builder(50, this.kernelSize)
                        .nIn(this.inChannels)
                        .stride(this.stride, this.stride)
                        .nOut(outChannels)
                        .padding(0, this.padding)
                        .activation(Activation.TANH)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);

        INDArray initialLayer = Nd4j.ones(1, 1, 50, 20);
        model.initializeLayers(initialLayer);
        List<INDArray> params = new ArrayList<>();
        params.add(biases);
        params.add(filters);
        model.setParams(Nd4j.toFlattened('f', params));
    }

    public INDArray forward(INDArray input) {
        model.initializeLayers(input);
        int[] shape = {1, 1, input.size(0), input.size(1)};
        INDArray i = Nd4j.createUninitialized(shape);
        for (int j = 0; j < input.size(0); j++) {
            INDArrayIndex[] index = {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(j), NDArrayIndex.all()};
            INDArrayIndex[] inputIndex = {NDArrayIndex.point(j), NDArrayIndex.all()};
            i.put(index, input.get(inputIndex));
        }
        INDArray output = model.output(i, false);
        output = output.reshape(output.size(1), output.size(3));
        return output;
    }
}
