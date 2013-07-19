package de.tuberlin.dima.ml.mapred;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.mapred.eval.EvalJob;
import de.tuberlin.dima.ml.mapred.logreg.ensemble.EnsembleJob;

public class EnsembleJobTest {
  
  private static final DatasetInfo DATASET = RCV1DatasetInfo.get();
  
  private static final int ENSEMBLE_SIZE = 4;
  
  // Currently we train a hardcoded single 1-vs-all classifier
  private static final String TARGET_POSITIVE = "CCAT";
  
  private static final String INPUT_FILE_TRAIN = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_train.seq";
//  private static final String INPUT_FILE_TRAIN = "rcv1-v2/lyrl2004_vectors_train_5000.seq";
  
  private static final String INPUT_FILE_TEST = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_test_20000.seq";
//  private static final String INPUT_FILE_TEST = "rcv1-v2/lyrl2004_vectors_test_5000.seq";
  
  private static final String OUTPUT_TRAIN_PATH = "output-aim3-ensemble";
  private static final String OUTPUT_TEST_PATH = "output-aim3-validation";

  @Test
  public void test() throws Exception {
    
    int labelDimension = DATASET.getLabelIdByName(TARGET_POSITIVE);
    
    ToolRunner.run(new EnsembleJob(
        INPUT_FILE_TRAIN, 
        OUTPUT_TRAIN_PATH, 
        ENSEMBLE_SIZE,
        labelDimension,
        (int)DATASET.getNumFeatures()), null);
    
    ToolRunner.run(new EvalJob(
        INPUT_FILE_TEST, 
        OUTPUT_TEST_PATH,
        OUTPUT_TRAIN_PATH,
        labelDimension), null);
  }

}
