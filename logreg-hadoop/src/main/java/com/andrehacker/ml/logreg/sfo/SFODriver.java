package com.andrehacker.ml.logreg.sfo;

import org.apache.hadoop.util.ToolRunner;

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
public class SFODriver {
  
  private String inputFileLocal;
  private String inputFileHdfs;
  private String outputPathTrain;
  private String outputPathTest;
  
  private int reducersTrain;
  private int reducersTest;
  
  private IncrementalModel baseModel;
  
  /**
   * Initializes a new empty base model
   */
  public SFODriver(String inputFileLocal,
      String inputFileHdfs,
      String outputPathTrain,
      String outputPathTest,
      int reducersTrain,
      int reducersTest) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPathTrain = outputPathTrain;
    this.outputPathTest = outputPathTest;
    this.reducersTrain = reducersTrain;
    this.reducersTest = reducersTest;
    
    // Create empty model
    baseModel = new IncrementalModel((int)GlobalJobSettings.datasetInfo.getVectorSize());
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
    SFOJobTools.writeBaseModel(baseModel);
    
    // Make current model available for Mappers of Train and Test job
    SFOJob jobTrain = new SFOJob(
        inputFileLocal,
        inputFileHdfs,
        outputPathTrain,
        reducersTrain);
    
    ToolRunner.run(jobTrain, null);
    
    System.out.println("Done training");

    ToolRunner.run(new EvalJob(
        inputFileLocal,
        inputFileHdfs,
        outputPathTest,
        reducersTest), null);
    System.out.println("Done validation");
  }

}
