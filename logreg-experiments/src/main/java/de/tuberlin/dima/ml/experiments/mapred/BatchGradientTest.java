package de.tuberlin.dima.ml.experiments.mapred;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.mapred.logreg.iterative.BatchGradientDriver;

public class BatchGradientTest {
  
  private static final DatasetInfo DATASET = RCV1DatasetInfo.get();
  
  private static final String TARGET_POSITIVE = "CCAT";

//  private static final String INPUT_FILE_TRAIN = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  private static final String INPUT_FILE_TRAIN ="hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  private static final String INPUT_FILE_TRAIN = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_train_10000.seq";
//  private static final String INPUT_FILE_TRAIN ="datasets/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  
  private static final String OUTPUT_TRAIN_PATH = "output-aim3-batchgd";
  
  private static final int MAX_ITERATIONS = 3;

  public static void main(String[] args) throws Exception {
    
    int labelDimension = DATASET.getLabelIdByName(TARGET_POSITIVE);

    double initial = 1;
    
    String hdfsAddress = "";
    String jobtrackerAddress = "";
    
    BatchGradientDriver bgDriver = new BatchGradientDriver(
        INPUT_FILE_TRAIN, 
        OUTPUT_TRAIN_PATH, 
        MAX_ITERATIONS,
        initial,
        labelDimension,
        (int)DATASET.getNumFeatures(),
        hdfsAddress,
        jobtrackerAddress);
    
    bgDriver.train();
  }
  
}