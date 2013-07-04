package com.andrehacker.ml.logreg.sfo;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Ignore;
import org.junit.Test;

import com.andrehacker.ml.datasets.DatasetInfo;
import com.andrehacker.ml.datasets.DonutDatasetInfo;
import com.andrehacker.ml.datasets.RCV1DatasetInfo;

public class SFOJobTest {
  
  
  static final String TRAIN_OUTPUT_PATH = "output-sfo-train";
  private static final String TEST_OUTPUT_PATH = "output-sfo-test";
  
  private static final int REDUCERS_TRAIN = 4;
  private static final int REDUCERS_TEST = 4;

  @Test
  @Ignore
  public void testDonut() throws Exception {
    
    GlobalJobSettings.datasetInfo = DonutDatasetInfo.get();
    String inputFileLocal = "/home/andre/dev/datasets/donut/donut.csv.seq";
    String inputFileHdfs = "donut/donut.csv.seq";
    
    runSFOTest(
        DonutDatasetInfo.get(),
        inputFileLocal, 
        inputFileHdfs,
        8
        );
  }

  @Test
  public void testRCV1() throws Exception {
    
    GlobalJobSettings.datasetInfo = RCV1DatasetInfo.get();
    String inputFileLocal = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_sfo_train_5000.seq";
    String inputFileHdfs = "rcv1-v2/lyrl2004_vectors_sfo_train_5000.seq";
    String predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt";
    
    RCV1DatasetInfo.readPredictorNames(predictorNamePath);
    
    runSFOTest(
        DonutDatasetInfo.get(),
        inputFileLocal, 
        inputFileHdfs,
        37664   // shar 
        );
  }
  
  private void runSFOTest(DatasetInfo datasetInfo, String inputFileLocal, String inputFileHdfs, int actualBestDim) throws Exception {
    
    // Initialize new empty base model
    SFODriver driver = new SFODriver(
        inputFileLocal,
        inputFileHdfs,
        TRAIN_OUTPUT_PATH,
        TEST_OUTPUT_PATH,
        REDUCERS_TRAIN,
        REDUCERS_TEST);
    
    int iterations = 1;
    for (int i=0; i<iterations; ++i ) {
      driver.runSFO();
      
      List<SFODriver.FeatureGain> gains = driver.getGains();
      printTopGains(gains);
      int bestDim = gains.get(0).getDimension();
      System.out.println("Best dimension: " + bestDim + " (" + GlobalJobSettings.datasetInfo.getFeatureName(bestDim) + ")");
      if (i==0)
        assertTrue(bestDim == actualBestDim);
      
      driver.addBestFeature();
      
      driver.retrainBaseModel();
    }
  }
  
  private void printTopGains(List<SFODriver.FeatureGain> gains) {
    for (int i=0; i<10 && i<gains.size(); ++i) {
      System.out.println("d " + gains.get(i).getDimension() + 
          " (" + GlobalJobSettings.datasetInfo.getFeatureName(gains.get(i).getDimension()) 
          + ") gain: " + gains.get(i).getGain());
    }
  }

}
