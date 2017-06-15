package io.anserini.qa.ranking;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.generic.GenericArray;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.DatumReader;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by jackzhang on 6/7/17.
 */
public class AvroLoader {

    private static final class Args {
        @Option(name = "-schema", metaVar = "[path]", required = true, usage = "schema path")
        String schema;

        @Option(name = "-avro", metaVar = "[path]", required = true, usage = "avro input")
        String avroFile;
    }

    public static ArrayList<INDArray> loadND4JArray(String schemaFile, String avroFile) throws IOException {
        Schema schema = new Schema.Parser().parse(new File(schemaFile));
        DatumReader<GenericRecord> datumReader = new GenericDatumReader<GenericRecord>(schema);
        File file = new File(avroFile);
        DataFileReader<GenericRecord> dataFileReader = new DataFileReader<GenericRecord>(file, datumReader);
        GenericRecord weightObject = null;
        ArrayList<INDArray> weightsArray = new ArrayList<INDArray>();

        while (dataFileReader.hasNext()) {
            weightObject = dataFileReader.next(weightObject);
            GenericArray<Integer> d = (GenericArray<Integer>)weightObject.get("dimension");
            int[] dimensions = new int[d.size()];
            int i = 0;
            for (int k : d) {
                dimensions[i] = k;
                i++;
            }

            GenericArray<Double> w = (GenericArray<Double>)weightObject.get("weights");
            double[] weights = new double[w.size()];
            i = 0;
            for (double j : w) {
                weights[i] = j;
                i++;
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
        final Args argv = new Args();
        CmdLineParser parse = new CmdLineParser(argv, ParserProperties.defaults().withUsageWidth(100));
        try {
            parse.parseArgument(args);
        } catch (CmdLineException e){
            System.err.println(e.getMessage());
        }

        String schemaFile = argv.schema;
        String avroFile = argv.avroFile;
        try {
            ArrayList<INDArray> result = loadND4JArray(schemaFile, avroFile);
            for (INDArray r : result) {
                System.out.println(r);
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }
}
