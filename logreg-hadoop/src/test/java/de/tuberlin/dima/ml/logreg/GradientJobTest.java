package de.tuberlin.dima.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.logreg.iterative.GradientJob;

public class GradientJobTest {
  
  private static final DatasetInfo DATASET = RCV1DatasetInfo.get();
  private static final String TARGET_POSITIVE = "CCAT";
  
//  private static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
  private static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_train_10000.seq";
//  private static final String INPUT_FILE_TRAIN_HDFS ="hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  private static final String OUTPUT_TRAIN_PATH = "output-aim3-gradient";

  @Test
  public void test() throws Exception {

    int labelDimension = DATASET.getLabelIdByName(TARGET_POSITIVE);
    
    GradientJob job = new GradientJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH,
        labelDimension,
        (int)DATASET.getNumFeatures());    

    ToolRunner.run(job, null);
  }
}
