package io.anserini.qa.ranking;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class SMModelTest {
    @Test
    public void testSMModelConvWithAvroLoader() {
        try {
            Nd4j.getRandom().setSeed(1234);
            String questionFeaturesPath = this.getClass().getResource("q1.csv").getPath();
            String answerFeaturesPath = this.getClass().getResource("a1.csv").getPath();
            String externalFeaturesPath = this.getClass().getResource("ext_1.csv").getPath();
            INDArray question = CSVLoader.load2DNd4jArray(questionFeaturesPath);
            INDArray answer = CSVLoader.load2DNd4jArray(answerFeaturesPath);
            INDArray externalFeatures = CSVLoader.load2DNd4jArray(externalFeaturesPath);

            String avroSchemaPath = this.getClass().getResource("weights.avsc").getPath();
            String avroWeightsPath = this.getClass().getResource("weights.avro").getPath();
            SMModel model = new SMModel(avroSchemaPath, avroWeightsPath);
            INDArray finalLayer = model.forward(question, answer, externalFeatures);

            assertEquals(String.valueOf(-7.61E-5), String.valueOf(Math.floor(finalLayer.getDouble(0) * 1e7) / 1e7));
            assertEquals(String.valueOf(-9.4843), String.valueOf(Math.floor(finalLayer.getDouble(1) * 1e4) / 1e4));
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }
}
