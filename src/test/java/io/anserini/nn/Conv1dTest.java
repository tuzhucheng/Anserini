package io.anserini.nn;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class Conv1dTest {
    @Test
    public void testConv1dSimple() {
        double[] rawInput = {10, 50, 60, 10, 20, 40, 30};
        INDArray input = Nd4j.create(rawInput);

        double[] rawFilter = {1.0 / 3, 1.0 / 3, 1.0 / 3};
        INDArray filter = Nd4j.create(rawFilter);

        double[] rawExpected = {40, 40, 30, 70.0 / 3, 30};
        INDArray expectedOutput = Nd4j.create(rawExpected);

        Conv1d conv = new Conv1d(1, 1, filter.shape()[1], 1, 0);
        INDArray actualOutput = conv.forward(input, filter);

        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testConv1dWithPadding() {
        double[] rawInput = {10, 50, 60, 10, 20, 40, 30};
        INDArray input = Nd4j.create(rawInput);

        double[] rawFilter = {1.0 / 3, 1.0 / 3, 1.0 / 3};
        INDArray filter = Nd4j.create(rawFilter);

        double[] rawExpected = {20, 40, 40, 30, 70.0 / 3, 30, 70.0 / 3};
        INDArray expectedOutput = Nd4j.create(rawExpected);

        Conv1d conv = new Conv1d(1, 1, filter.shape()[1], 1, 1);
        INDArray actualOutput = conv.forward(input, filter);

        assertEquals(expectedOutput, actualOutput);
    }
}
