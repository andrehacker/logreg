package de.tuberlin.dima.ml.logreg.sequential;

import java.io.IOException;

import org.junit.Test;

import de.tuberlin.dima.ml.logreg.sequential.LogRegSGDRCV1;

public class LogRegSGDRCV1Test {

  @Test
  public void testTrainRCV1() throws IOException {
    String trainingFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.dat";
    String testFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_pt0.dat";
    
    LogRegSGDRCV1 lr = new LogRegSGDRCV1();
    lr.trainRCV1(trainingFile, testFile);
  }

}
