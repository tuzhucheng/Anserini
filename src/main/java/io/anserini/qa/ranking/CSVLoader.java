package io.anserini.qa.ranking;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CSVLoader {

    public static INDArray load2DNd4jArray(String csvFile) throws IOException {
        BufferedReader br = null;
        String line = null;
        ArrayList<ArrayList<Double>> floats = new ArrayList<ArrayList<Double>>();
        int innerDimension = 0;

        br = new BufferedReader(new FileReader(csvFile));
        while ((line = br.readLine()) != null) {
            // use comma as separator
            String[] parts = line.split(",");
            ArrayList<Double> rowFloats = new ArrayList<Double>();
            innerDimension = parts.length;
            for (String p : parts) {
                rowFloats.add(Double.valueOf(p));
            }
            floats.add(rowFloats);
        }

        double[][] matrix = new double[floats.size()][innerDimension];
        for (int i = 0; i < floats.size(); i++) {
            for (int j = 0; j < innerDimension; j++) {
                matrix[i][j] = floats.get(i).get(j);
            }
        }

        INDArray nd4jArray = Nd4j.create(matrix);
        return nd4jArray;
    }

    public static INDArray load1DNd4jArray(String csvFile) throws IOException {
        BufferedReader br = null;
        String line = null;
        ArrayList<Double> floats = new ArrayList<Double>();

        br = new BufferedReader(new FileReader(csvFile));
        while ((line = br.readLine()) != null) {
            floats.add(Double.valueOf(line));
        }

        double[] temp = new double[floats.size()];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = floats.get(i);
        }

        INDArray nd4jArray = Nd4j.create(temp);
        return nd4jArray;
    }
}
