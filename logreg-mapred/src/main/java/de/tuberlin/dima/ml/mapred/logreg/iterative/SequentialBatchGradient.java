package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.mapred.writables.IDAndLabels;
import de.tuberlin.dima.ml.validation.OnlineAccuracy;

public class SequentialBatchGradient {

  private static final int TARGET = 0; // CCAT
  
  private List<Vector> trainingInput, trainingLabel;
  private List<Vector> testInput, testLabel;

  private int numFeatures;
  private int maxIterations;
  
  private static double bestAccuracy = 0;
  private static int bestIteration = 0;

  public static void main(String[] args) {

//    String pathToTrainingFile = "/Users/uce/Desktop/rcv1/vectors/lyrl/lyrl2004_vectors_train.seq";
//    String pathToTestFile = "/Users/uce/Desktop/rcv1/vectors/lyrl/lyrl2004_vectors_test.seq";
    String pathToTrainingFile = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_train.seq";
    String pathToTestFile = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_test_20000.seq";

    int maxIterations = 100;

    SequentialBatchGradient bgd = new SequentialBatchGradient(pathToTrainingFile, pathToTestFile, maxIterations);

    double learningRate = 0.04;
    double regularization = 0.5;
    @SuppressWarnings("unused")
    Vector model = bgd.train(learningRate, regularization);
    
    System.out.println("Best accuracy=" + bestAccuracy + " (in iteration " + bestIteration + ")");

//    bgd.evaluate(model, 100);
  }

  public SequentialBatchGradient(String pathToTrainingFile, String pathToTestFile, int maxIterations) {

    System.out.println("new bgd instance with");
    System.out.println(" * maxIterations: " + maxIterations);

    // inputs
    this.trainingInput = new ArrayList<Vector>();
    this.trainingLabel = new ArrayList<Vector>();

    this.testInput = new ArrayList<Vector>();
    this.testLabel = new ArrayList<Vector>();

    System.out.println();
    System.out.print("reading training file into memory");

    // read training file
    this.numFeatures = read(pathToTrainingFile, this.trainingInput, this.trainingLabel);

    System.out.println("done");
    System.out.println("training vectors: " + this.trainingInput.size());

    System.out.println();

    System.out.print("reading test file into memory");

    // read test file
    read(pathToTestFile, this.testInput, this.testLabel);

    System.out.println("done");
    System.out.println("test vectors: " + this.testInput.size());

    System.out.println();

    System.out.println("num features: " + this.numFeatures);

    // max iterations
    this.maxIterations = maxIterations;
  }

  private int read(String pathToFile, List<Vector> input, List<Vector> labels) {

    Path path = new Path(pathToFile);
    Configuration conf = new Configuration();

    int numFeatures = 0;

    int numRead = 0;
    for (Pair<IDAndLabels, VectorWritable> in : new SequenceFileIterable<IDAndLabels, VectorWritable>(
        path, conf)) {
      labels.add(in.getFirst().getLabels());
      input.add(new SequentialAccessSparseVector(in.getSecond().get()));

      if (numFeatures == 0)
        numFeatures = in.getSecond().get().size();

      numRead++;

      if (numRead % 5000 == 0)
        System.out.print(".");
    }

    return numFeatures;
  }

  public Vector train(double learningRate, double regularization) {
    System.out.println();
    System.out.println("training model on training set with learning rate of " + learningRate);

    Vector model = new DenseVector(this.numFeatures);

    for (int iteration = 0; iteration < this.maxIterations; iteration++) {
      
      System.out.print("iteration " + (iteration+1));
      Vector gradient = new DenseVector(this.numFeatures);
      
      for (int i = 0; i < this.trainingInput.size(); i++) {

        Vector x = this.trainingInput.get(i);
        double y = this.trainingLabel.get(i).get(TARGET);

        Vector partial = LogRegMath.computePartialGradient(x, model, y);
        gradient.assign(partial, Functions.PLUS);
        
        if (i % 5000 == 0)
          System.out.print(".");
      }
      System.out.println("done");
      
//      gradient.assign(Functions.DIV, trainingSize);
      
      Vector regularizationAdd = new DenseVector(this.numFeatures);
      regularizationAdd = model.times(regularization);
      
      gradient.assign(regularizationAdd, Functions.PLUS);
      
      gradient.assign(Functions.MULT, learningRate);
      
      // weight update: w = w - 1/N * \gamma * gradient
      model.assign(gradient, Functions.MINUS);
      
      if ((iteration+1) % 5 == 0) {
        evaluate(model, iteration);
        System.out.println();
      }
    }

    System.out.println();
    System.out.println("trained model with " + model.getNumNonZeroElements() + " non zero elements");

    return model;
  }

  private double evaluate(Vector model, int iteration) {
    OnlineAccuracy evaluation = new OnlineAccuracy(0.5);

    System.out.println();
    System.out.print("evaluating model on test set");
    for (int i = 0; i < this.testInput.size(); i++) {

      Vector x = this.testInput.get(i);

      int actualTarget = (int) this.testLabel.get(i).get(TARGET);
      double prediction = LogRegMath.predict(x, model);

      evaluation.addSample(actualTarget, prediction);

      if (i % 5000 == 0)
        System.out.print(".");
    }
    System.out.println();

    double accuracy = evaluation.getAccuracy();
    if (accuracy > bestAccuracy) {
      bestAccuracy = accuracy;
      bestIteration = iteration;
    }

    System.out.println("accuracy: " + accuracy);

    return accuracy;
  }
}