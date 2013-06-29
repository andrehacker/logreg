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
  private String jarPath;
  private String configFilePath;
  private String trainOutputPath;
  private String testOutputPath;
  private int trainReducers;
  private int testReducers;
  
  public SFODriver(String inputFileLocal,
      String inputFileHdfs,
      String trainOutputPath,
      String testOutputPath,
      String jarPath,
      String configFilePath,
      int trainReducers,
      int testReducers) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.trainOutputPath = trainOutputPath;
    this.testOutputPath = testOutputPath;
    this.jarPath = jarPath;
    this.configFilePath = configFilePath;
    this.trainReducers = trainReducers;
    this.testReducers = testReducers;
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
    
    ToolRunner.run(new SFOJob(
        inputFileLocal, 
        inputFileHdfs, 
        trainOutputPath, 
        jarPath, 
        configFilePath, 
        trainReducers), null);
    
    System.out.println("Done training");

    ToolRunner.run(new EvalJob(
        inputFileLocal,
        inputFileHdfs,
        testOutputPath,
        jarPath,
        configFilePath,
        testReducers), null);
    System.out.println("Done validation");
    
  }

}
