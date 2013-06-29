package com.andrehacker.ml.logreg.sfo;

import org.junit.Test;

public class SFOJobTest {
  
  private static final String INPUT_FILE_LOCAL = "/home/andre/dev/datasets/donut/donut.csv.seq";
  private static final String INPUT_FILE_HDFS = "donut/donut.csv.seq";
  
  // TODO Make this available any other way for EvalMapper
  static final String TRAIN_OUTPUT_PATH = "output-sfo-train";
  private static final String TEST_OUTPUT_PATH = "output-sfo-test";
  
  private static final String JAR_PATH = "target/logreg-0.0.1-SNAPSHOT-job.jar";
  private static final String CONFIG_FILE_PATH = "core-site.xml";
  
  private static final int REDUCERS_TRAIN = 4;
  private static final int REDUCERS_TEST = 4;

  @Test
  public void test() throws Exception {
    
    SFODriver driver = new SFODriver(
        INPUT_FILE_LOCAL, 
        INPUT_FILE_HDFS, 
        TRAIN_OUTPUT_PATH, 
        TEST_OUTPUT_PATH, 
        JAR_PATH, 
        CONFIG_FILE_PATH, 
        REDUCERS_TRAIN, 
        REDUCERS_TEST);
    
    driver.runSFO();
    
    // TODO Postprocess: Find best model (highest gain)
    
    // TODO Add best feature and retrain whole model
    
  }

}
