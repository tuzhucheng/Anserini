package io.anserini.qa.ranking;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import weightList.*;

/**
 * Created by jackzhang on 9/7/17.
 */

public class ThriftLoader {
    private static final class Args {
        // Code generation, no need for passing in schema file.
        @Option(name = "-thrift", metaVar = "[path]", required = true, usage = "thrift input")
        String thriftFile;
    }

    public static ArrayList<INDArray> loadND4JArray(String thriftFile) throws IOException, TException {
        File file = new File(thriftFile);
        ArrayList<INDArray> weightsArray = new ArrayList<INDArray>();

        // File input stream.
        FileInputStream fis = new FileInputStream(file);
        byte[] byteArray = IOUtils.toByteArray(fis);
        TDeserializer deserializer = new TDeserializer(new TBinaryProtocol.Factory());

        WeightList weightList = new WeightList();
        deserializer.deserialize(weightList, byteArray);

        for (int i = 0; i < weightList.getListNum(); i++) {

            // Convert list to array.
            List<Integer> dimensions_list = weightList.weightList.get(i).dimension;
            int[] dimensions = new int[dimensions_list.size()];
            int j = 0;
            for (int k : dimensions_list) {
                dimensions[j] = k;
                j++;
            }

            // Convert list to array.
            List<Double> w = weightList.weightList.get(i).weights;
            double[] weights = new double[w.size()];
            j = 0;
            for (double k : w) {
                weights[j] = k;
                j++;
            }

            INDArray nd4jArray = Nd4j.create(weights);
            switch (dimensions.length) {
                case 2: nd4jArray = nd4jArray.reshape(dimensions[0], dimensions[1]);
                    break;
                case 3: nd4jArray = nd4jArray.reshape(dimensions[0], dimensions[1], dimensions[2]);
                    break;
            }
            weightsArray.add(nd4jArray);
        }
        return weightsArray;
    }

    public static void main(String[] args) {
        final ThriftLoader.Args argv = new ThriftLoader.Args();
        CmdLineParser parse = new CmdLineParser(argv, ParserProperties.defaults().withUsageWidth(100));
        try {
            parse.parseArgument(args);
        } catch (CmdLineException e){
            System.err.println(e.getMessage());
        }

        String thriftFile = argv.thriftFile;
        try {
            ArrayList<INDArray> result = loadND4JArray(thriftFile);
            for (INDArray r : result) {
                System.out.println(r);
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
        } catch(TException e){
            System.err.println(e.getMessage());
        }
    }
}
