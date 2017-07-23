package io.anserini.qa.ranking;

import io.anserini.embeddings.TermNotFoundException;
import io.anserini.embeddings.WordEmbeddingDictionary;
import org.kohsuke.args4j.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class Bridge {
  private int vocabSize;
  private int vectorDimension;
  private SMModel model;
  private Map<String, Integer> vocabDictionary;
  private INDArray unknownVector;
  private WordEmbeddingDictionary wordEmbeddingDictionary;

  public static final class Args {
    // required arguments
    @Option(name = "-index", metaVar = "[Path]", required = true, usage = "index path")
    public String index;

    @Option(name = "-w2vCache", metaVar = "[Path]", required = true, usage = "Word embedding cache file")
    public String w2vCacheFile;

    @Option(name = "-model", metaVar = "[Path]", required = true, usage = "path to the model weights")
    public String model;

    @Option(name = "-dataset", metaVar = "[Path]", required = true, usage = "path of the TrecQA dataset folder")
    public String dataset;

    @Option(name = "-output", metaVar = "[Path]", required = true, usage = "output file path")
    public String output = "run";
  }

  public Bridge(String index, String w2vCache, String weightsPath) throws IOException {
    Nd4j.getRandom().setSeed(1234);
    String avroSchemaPath = this.getClass().getResource("weights.avsc").getPath();
    model = new SMModel(avroSchemaPath, weightsPath);

    this.vocabDictionary = new HashMap<>();
    wordEmbeddingDictionary = new WordEmbeddingDictionary(index);
    preloadCachedEmbeddings(w2vCache);

  }

  public void preloadCachedEmbeddings(String w2vCache) throws IOException {
    List<String> lines = Files.readAllLines(Paths.get(w2vCache + ".dimensions"));
    String[] sizeDimension = lines.get(0).trim().split("\\s+");
    vocabSize = Integer.parseInt(sizeDimension[0]);
    vectorDimension = Integer.parseInt(sizeDimension[1]);

//    ToDo: initialize W
    try(BufferedReader br = new BufferedReader(new FileReader(w2vCache + ".vocab"))) {
      int i = 0;
      String line;

      while ((line = br.readLine()) != null) {
        vocabDictionary.put(line.trim(), i);
        i++;
      }
      Random rng = Nd4j.getRandom();
      unknownVector = Nd4j.rand(1, vectorDimension, -0.25, 0.25, rng);
    }
  }


  public INDArray makeInputMatrix(String sentence) throws IOException {
    String[] terms = sentence.trim().split("\\s+");
    String[] reducedTerms = Arrays.copyOfRange(terms, 0, Math.min(60, terms.length));
    INDArray sentenceEmbedding = Nd4j.zeros(50, reducedTerms.length);

    for (int i = 0; i < reducedTerms.length; i++) {
      INDArrayIndex columnSlice[] = { NDArrayIndex.all(), NDArrayIndex.point(i) };
      String term = reducedTerms[i];

      INDArray wordVector = null;
      if (vocabDictionary.keySet().contains(term)) {
        try {
          wordVector = Nd4j.create(wordEmbeddingDictionary.getEmbeddingVector(term));
        } catch (ArrayIndexOutOfBoundsException e) {
          System.out.println(term + " is in dimensions but not in index.");
          wordVector = unknownVector;
        } catch (TermNotFoundException e) {
          e.printStackTrace();
        }
      } else {
        wordVector = unknownVector;
      }
      sentenceEmbedding.put(columnSlice, wordVector);
    }
    return sentenceEmbedding;
  }


  public Map<String, Double> rerankCandidates(String question, List<String> answers, String index) throws Exception {
    Map<String, Double> sentScore = new HashMap<>();
    FeaturePreparer pF = new FeaturePreparer(index);
    INDArray questionEmbedding = makeInputMatrix(question);

    for (String answer : answers) {
      double overlap = pF.computeOverlap(question, answer, false);
      double idfOverlap = pF.idfWeightedOverlap(question, answer, false);
      double overlapStopWords = pF.computeOverlap(question, answer, true);
      double idfOverlapStopWords = pF.idfWeightedOverlap(question, answer, true);

      double[] rawExternalFeatures = {overlap, idfOverlap, overlapStopWords, idfOverlapStopWords};
      INDArray externalFeatures = Nd4j.create(rawExternalFeatures);

      // TODO Batching
      INDArray answerEmbedding = makeInputMatrix(answer);
      INDArray logPreds = model.forward(questionEmbedding, answerEmbedding, externalFeatures);
      double pred = Math.exp(logPreds.getDouble(1));
      sentScore.put(answer, pred);
    }

    return sentScore;
  }


  public static void main(String[] args) throws Exception {
    Bridge.Args bridgeArgs = new Bridge.Args();
    CmdLineParser parser = new CmdLineParser(bridgeArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ Bridge.class.getSimpleName() +
              parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }
    Bridge br = new Bridge(bridgeArgs.index, bridgeArgs.w2vCacheFile, bridgeArgs.model);

    String[] config = {"raw-dev", "raw-test"};

    long totalElapsed = 0;
    int count = 0;
    for (String split : config) {
      BufferedReader questionFile = new BufferedReader(new FileReader(bridgeArgs.dataset + "/" + split + "/a.toks"));
      BufferedReader answerFile = new BufferedReader(new FileReader(bridgeArgs.dataset + "/" + split + "/b.toks"));
      BufferedReader idFile = new BufferedReader(new FileReader(bridgeArgs.dataset + "/" + split + "/id.txt"));

      BufferedWriter outputFile = new BufferedWriter(new FileWriter(bridgeArgs.output + ".TrecQA." + split + ".txt"));

      String old_id = "0";
      String previousQuestion = "";
      List<String> answerList = new ArrayList<>();

      while (true) {
        String question = questionFile.readLine();
        String answer = answerFile.readLine();
        String id = idFile.readLine();

        if (question == null || answer == null || id == null) {
          break;
        }

        // collect all the answers correponding to an id and rerank
        if (!old_id.equals(id) && !old_id.equals("0")) {
          long start = System.nanoTime();
          Map<String, Double> sentenceScore = br.rerankCandidates(previousQuestion, answerList, bridgeArgs.index);
          long end = System.nanoTime();
          long elapsed = end - start;
          totalElapsed += elapsed;
          count += 1;

          // 32.1 0 1 0 0.13309887051582336 smmodel
          int i = 0;
          for (Map.Entry<String, Double> cand : sentenceScore.entrySet()) {
            outputFile.write(old_id + " Q0 " + i + " 0 " + cand.getValue() + " smmodel\n");
            i++;
          }
          answerList.clear();
        }

        previousQuestion = question;
        answerList.add(answer);
        old_id = id;
      }

      outputFile.close();
    }
    long totalElapsedSeconds = TimeUnit.NANOSECONDS.toSeconds(totalElapsed);
    System.out.println("Elapsed time (s): " + totalElapsedSeconds);
    System.out.println("Questions: " + count);
    System.out.println("QPS: " + count * 1.0 / totalElapsedSeconds);
  }

}
