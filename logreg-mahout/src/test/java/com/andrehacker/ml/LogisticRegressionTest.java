package com.andrehacker.ml;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.Lists;

public class LogisticRegressionTest {

  @Test
  public void test() {
    List<String> predictorNames = Lists.newArrayList(new String[] {
      //"x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c", "bias"
      //"x", "y", "shape", "a", "b", "c"
      "x", "y", "a", "b", "c"    // like Mahout page 252
     });
     
     double initialWeight = 0;
     LogisticRegression logTrainer = new LogisticRegression();
     try {
       logTrainer.trainBatchGD("donut.csv", predictorNames, 2000, 0.05, initialWeight);
     } catch (IOException e1) {
       e1.printStackTrace();
     }
  }

}
