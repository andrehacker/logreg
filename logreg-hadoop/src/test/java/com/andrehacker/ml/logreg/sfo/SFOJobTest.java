package com.andrehacker.ml.logreg.sfo;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.andrehacker.ml.logreg.sfo.SFOJob;

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
    
    ToolRunner.run(new SFOJob(
        INPUT_FILE_LOCAL, 
        INPUT_FILE_HDFS, 
        TRAIN_OUTPUT_PATH, 
        JAR_PATH, 
        CONFIG_FILE_PATH, 
        REDUCERS_TRAIN), null);
    
    System.out.println("Done training");

    ToolRunner.run(new EvalJob(
        INPUT_FILE_LOCAL,
        INPUT_FILE_HDFS,
        TEST_OUTPUT_PATH,
        JAR_PATH,
        CONFIG_FILE_PATH,
        REDUCERS_TEST), null);
    System.out.println("Done validation");
  }

}
