package de.tuberlin.dima.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.logreg.iterative.GradientJob;
import de.tuberlin.dima.ml.logreg.iterative.LBFGSDriver;
import de.tuberlin.dima.ml.logreg.iterative.TrainingErrorJob;

public class LBFGSJobTest {

  private static final DatasetInfo DATASET = RCV1DatasetInfo.get();
  private static final String TARGET_POSITIVE = "CCAT";

//  private static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
  private static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  private static final String INPUT_FILE_TRAIN_HDFS = "hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";

  private static final String OUTPUT_TRAIN_PATH = "output-aim3-lbfgs";

  @Test
  public void test() throws Exception {

    int labelDimension = DATASET.getLabelIdByName(TARGET_POSITIVE);

    TrainingErrorJob trainingErrorJob = new TrainingErrorJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH,
        labelDimension);

    GradientJob gradientJob = new GradientJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH,
        labelDimension,
        (int)DATASET.getNumFeatures());

    double eps = 0.1;

    double[] initial = new double[(int) RCV1DatasetInfo.get().getNumFeatures()];

    LBFGSDriver lbfgs = new LBFGSDriver(trainingErrorJob, gradientJob, eps, initial);

    ToolRunner.run(lbfgs, null);
  }
}