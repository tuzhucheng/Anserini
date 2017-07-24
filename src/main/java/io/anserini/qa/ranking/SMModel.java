package io.anserini.qa.ranking;

import io.anserini.nn.Conv1d;
import io.anserini.nn.DL4JConv1d;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;

public class SMModel {
    private INDArray questionConvFilters;
    private INDArray answerConvFilters;
    private INDArray questionConvFilterBiases;
    private INDArray answerConvFilterBiases;
    private INDArray hiddenLayerWeights;
    private INDArray hiddenLayerBiases;
    private INDArray softmaxLayerWeights;
    private INDArray softmaxLayerBiases;
    private DL4JConv1d questionConv;
    private DL4JConv1d answerConv;

    public SMModel(String avroWeightsSchema, String avroWeights) throws IOException {
        ArrayList<INDArray> weights = AvroLoader.loadND4JArray(avroWeightsSchema, avroWeights);
        this.questionConvFilters = weights.get(0);
        this.questionConvFilterBiases = weights.get(1);
        this.answerConvFilters = weights.get(2);
        this.answerConvFilterBiases = weights.get(3);
        this.hiddenLayerWeights = weights.get(4).transposei();
        this.hiddenLayerBiases = weights.get(5);
        this.softmaxLayerWeights = weights.get(6).transposei();
        this.softmaxLayerBiases = weights.get(7);

        questionConv = new DL4JConv1d(1, questionConvFilters.size(0), 5, 1, 4, questionConvFilters, questionConvFilterBiases);
        answerConv = new DL4JConv1d(1, answerConvFilters.size(0), 5, 1, 4, answerConvFilters, answerConvFilterBiases);
    }

    public INDArray forward(INDArray question, INDArray answer, INDArray externalFeatures) {
        // Convolution
        INDArray questionConvFeatureMaps = questionConv.forward(question);
        INDArray answerConvFeatureMaps = answerConv.forward(answer);

        // Pooling
        INDArray questionPooled = Nd4j.max(questionConvFeatureMaps, 1);
        INDArray answerPooled = Nd4j.max(answerConvFeatureMaps, 1);

        // Join layer
        INDArray joinLayer = Nd4j.zeros(204);
        INDArrayIndex questionIndex[] = { NDArrayIndex.interval(0, 100) };
        INDArrayIndex answerIndex[] = { NDArrayIndex.interval(100, 200) };
        INDArrayIndex extFeatsIndex[] = { NDArrayIndex.interval(200, 204) };
        joinLayer.put(questionIndex, questionPooled);
        joinLayer.put(answerIndex, answerPooled);
        joinLayer.put(extFeatsIndex, externalFeatures);

        // Hidden layer
        INDArray hiddenLayer = joinLayer.mmul(hiddenLayerWeights).addi(hiddenLayerBiases);
        Nd4j.getExecutioner().exec(new Tanh(hiddenLayer));

        // Softmax
        INDArray finalLayer = hiddenLayer.mmul(softmaxLayerWeights).addi(softmaxLayerBiases);
        Nd4j.getExecutioner().exec(new LogSoftMax(finalLayer));
        return finalLayer;
    }

}
