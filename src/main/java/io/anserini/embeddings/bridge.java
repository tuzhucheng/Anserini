package io.anserini.embeddings;

import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class bridge {
  private int vocabSize;
  private int vectorDimension;
  private Map<Integer, String> vocabDictionary;
  private Map<String, INDArray> wordEmbedding;

  public static final class Args {
    // required arguments
    @Option(name = "-index", metaVar = "[Path]", required = true, usage = "index path")
    public String index;

    @Option(name = "-w2vCache", metaVar = "[Path]", required = true, usage = "Word embedding cache file")
    public String w2vCacheFile;

    @Option(name = "-model", metaVar = "[Path]", required = true, usage = "model path")
    public String model;
  }

  public bridge(Path index, Path w2vCache, Path model) throws IOException {
//    ToDo: initialize torch seed
//    ToDo: load model

    this.vocabDictionary = new HashMap();
    preloadCachedEmbeddings(w2vCache);


  }

  public void preloadCachedEmbeddings(Path w2vCache) throws IOException {
    List<String> lines = Files.readAllLines(w2vCache);
    String[] sizeDimension = lines.get(0).trim().split("\\t");
    this.vocabSize = Integer.parseInt(sizeDimension[0]);
    this.vectorDimension = Integer.parseInt(sizeDimension[1]);

//    ToDo: initialize W
    try(BufferedReader br = new BufferedReader(new FileReader(w2vCache + ".vocab"))) {
      int i = 0;
      String line = br.readLine();

      while (line != null) {
        line = br.readLine();
        vocabDictionary.put(i, line.trim());
        i++;
      }
    }
  }


  public void makeInputMatrix(String sentence) {
    String[] terms = sentence.trim().split("\\t");
    String[] reducedTerms = Arrays.copyOfRange(terms, 0, 60);

    for (String term : reducedTerms) {
      if (vocabDictionary.keySet().contains(term)) {

      } else {

      }
    }


  }

}
