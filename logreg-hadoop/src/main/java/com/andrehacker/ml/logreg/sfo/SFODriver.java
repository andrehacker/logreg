package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.andrehacker.ml.GlobalSettings;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

/**
 * Implements the Single Feature Optimization Algorithm as proposed by Singh et al. [1]
 * 
 * Internally uses Hadoop Jobs
 * 
 * REFERENCES
 * [1] Singh, S., Kubica, J., Larsen, S., & Sorokina, D. (2009).
 * Parallel Large Scale Feature Selection for Logistic Regression.
 * Optimization, 1172–1183.
 * Retrieved from http://www.additivegroves.net/papers/fslr.pdf
 * 
 * @author Andre Hacker
 *
 */
public class SFODriver extends Configured implements Tool {
  
  private String inputFile;
  
  private String trainOutputPath;
  private String testOutputPath;
  
  private int reducersTrain;
  private int reducersTest;
  
  private int numFeatures;
  
  private IncrementalModel baseModel;
  
  private List<FeatureGain> gains = Lists.newArrayList();
  
  /**
   * Initializes a new empty base model
   */
  public SFODriver(
      String inputFile,
      String outputPathTrain,
      String outputPathTest,
      int reducersTrain,
      int reducersTest,
      int numFeatures) {
    this.inputFile = inputFile;
    this.trainOutputPath = outputPathTrain;
    this.testOutputPath = outputPathTest;
    this.reducersTrain = reducersTrain;
    this.reducersTest = reducersTest;
    this.numFeatures = numFeatures;
    
    // Create empty model
    baseModel = new IncrementalModel(numFeatures);
  }

  /**
   * Whenever we are started through
   */
  @Override
  public int run(String[] args) throws Exception {
    if (args.length != 5) {
      System.err.printf("Usage: %s [generic options] <input> <outputTrain> <outputTest> <reducersTrain> <reducersTest>", getClass().getSimpleName());
      ToolRunner.printGenericCommandUsage(System.err);
      return -1;
    }
    return 0;
  }
  
  /**
   * Runs a single SFO-Iteration.
   * Computes the gain in metric (e.g. log-likelihood)
   * for all possible models with one more feature added 
   * to the current base model
   * 
   * Does not add the best feature to the base model,
   * allows the client to decide whether to add it or not.
   */
  public void runSFO() throws Exception {
    
    // Make base model available for train/test mappers
    SFOTools.writeBaseModel(baseModel);
    
    // ----- TRAIN -----
    ToolRunner.run(new SFOJob(
        inputFile,
        trainOutputPath,
        reducersTrain), null);
    
    System.out.println("Done training");
    
    // ----- TEST -----
    ToolRunner.run(new EvalJob(
        inputFile,
        testOutputPath,
        reducersTest,
        numFeatures,
        trainOutputPath), null);
    System.out.println("Done validation");
    
    // ----- Postprocess -----
    readGains();
  }

  public void addBestFeature() throws IOException {
    // Read coefficients
    // TODO Optimization: Don't read all into memory, just search the single coefficient
    Configuration conf = new Configuration();
    conf.addResource(new Path(GlobalSettings.CONFIG_FILE_PATH));
    List<Double> coefficients = SFOTools.readTrainedCoefficients(conf, numFeatures, trainOutputPath);
    
    // Add best to base model
    int bestDimension = gains.get(0).getDimension();
    baseModel.addDimensionToModel(bestDimension, coefficients.get(bestDimension));
    
    // Write updated model to hdfs
    SFOTools.writeBaseModel(baseModel);
    
    System.out.println("Added dimension " + bestDimension + " to base model with coefficient: " + coefficients.get(bestDimension));
    System.out.println("- New base model: " + baseModel.getW().toString());
  }
  
  public void retrainBaseModel() {
    // TODO Major: Retrain Base Model!
    System.out.println("Retraining base model not yet implemented");
  }
  
  private void readGains() throws IOException {
    // Read results from hdfs into memory
    gains = SFOTools.readEvalResult(testOutputPath);
    
    // Sort by gain
    Collections.sort(gains, Collections.reverseOrder());
  }
  
  public List<FeatureGain> getGains() {
    return gains;
  }
  
  static class FeatureGain implements Comparable<FeatureGain> {
    private int dimension;
    private double gain;    // currently log-likelihood gain
    
    public FeatureGain(int dimension, double gain) {
      this.dimension = dimension;
      this.gain = gain;
    }
    
    @Override
    public int compareTo(FeatureGain other) {
      return Doubles.compare(this.gain, other.gain);
    }
    public int getDimension() {
      return dimension;
    }
    public double getGain() {
      return gain;
    }
  }

}
