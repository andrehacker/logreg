package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.util.ArrayList;
import java.util.Arrays;
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
import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

/**
 * Sequential implementation of L-BFGS, using Stanford 
 * 
 * Reads whole training set into main memory
 */
public class SequentialLBFGS implements DiffFunction {

  private static final int TARGET = 0; // CCAT

  private List<Vector> trainingInput, trainingLabel;
  private List<Vector> testInput, testLabel;

  private int numFeatures;
  private double[] initialWeights;
  private double eps;
  private int maxIterations;

  private QNMinimizer minimizer;

  public static void main(String[] args) {

//    String pathToTrainingFile = "/Users/uce/Desktop/rcv1/vectors/lyrl/lyrl2004_vectors_train.seq";
//    String pathToTestFile = "/Users/uce/Desktop/rcv1/vectors/lyrl/lyrl2004_vectors_test.seq";
    String pathToTrainingFile = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_train.seq";
    String pathToTestFile = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_test_20000.seq";


    double initialWeight = 0;
    double eps = 0.01;
    int maxIterations = -1;

    SequentialLBFGS lbfgs = new SequentialLBFGS(pathToTrainingFile, pathToTestFile, initialWeight, eps, maxIterations);

    Vector model = lbfgs.train();

    lbfgs.evaluate(model);
  }

  public SequentialLBFGS(String pathToTrainingFile, String pathToTestFile,
      double initialWeight, double eps, int maxIterations) {

    System.out.println("new lbfgs instance with");
    System.out.println(" * initialWeight: " + initialWeight);
    System.out.println(" * eps: " + eps);
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

    // QNMinimizer
    this.initialWeights = new double[this.numFeatures];
    Arrays.fill(this.initialWeights, initialWeight);

    this.eps = eps;

    this.maxIterations = maxIterations;

    this.minimizer = new QNMinimizer(15, true);
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

  public Vector train() {
    System.out.println();
    System.out.print("training model on training set");

    Vector model = new DenseVector(this.numFeatures);

    double[] weights;

    if (this.maxIterations > 0) {
      weights = this.minimizer.minimize(this, this.eps, this.initialWeights, this.maxIterations);
    } else {
      // minimize until convergence
      weights = this.minimizer.minimize(this, this.eps, this.initialWeights);
    }

    model.assign(weights);

    System.out.println();
    System.out.println("trained model with " + model.getNumNonZeroElements() + " non zero elements");

    return model;
  }

  private double evaluate(Vector model) {
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

    System.out.println("accuracy: " + accuracy);

    return accuracy;
  }

  @Override
  public double valueAt(double[] weights) {

    Vector w = new DenseVector(weights);
    double trainingError = 0.0;

    for (int i = 0; i < this.trainingInput.size(); i++) {

      Vector x = this.trainingInput.get(i);
      double y = this.trainingLabel.get(i).get(TARGET);

      // TODO Probably we have another error function here
      trainingError += LogRegMath.computeSqError(x, w, y);
//      trainingError -= LogRegMath.logLikelihood(y, LogRegMath.predict(x, w));
    }

    return trainingError;
  }

  @Override
  public int domainDimension() {

    return this.numFeatures;
  }

  @Override
  public double[] derivativeAt(double[] weights) {

    Vector w = new DenseVector(weights);
    Vector gradient = new DenseVector(weights.length);

    // Add partial gradients
    for (int i = 0; i < this.trainingInput.size(); i++) {

      Vector x = this.trainingInput.get(i);
      double y = this.trainingLabel.get(i).get(TARGET);

      Vector partial = LogRegMath.computePartialGradient(x, w, y);
      gradient.assign(partial, Functions.PLUS);
    }

    // Vector to double array
    double[] d = new double[weights.length];
    for (int i = 0; i < gradient.size(); i++) {
      d[i] = gradient.get(i);
    }

    return d;
  }
}