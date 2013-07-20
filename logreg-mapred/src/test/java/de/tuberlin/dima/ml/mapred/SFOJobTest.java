package de.tuberlin.dima.ml.mapred;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Ignore;
import org.junit.Test;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.DonutDatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.mapred.logreg.sfo.SFODriver;

public class SFOJobTest {
  
  private static final String TRAIN_OUTPUT_PATH = "output-sfo-train";
  private static final String TEST_OUTPUT_PATH = "output-sfo-test";
  
  private static final int REDUCERS_TRAIN = 4;
  private static final int REDUCERS_TEST = 4;
  
  private static final int ITERATIONS = 2;

  @Test
  @Ignore
  public void testDonut() throws Exception {
    
    String inputFile = "/home/andre/dev/datasets/donut/donut.csv.seq";
//    String inputFile = "donut/donut.csv.seq";
    
    runSFOTest(
        DonutDatasetInfo.get(),
        inputFile,
        8
        );
  }

  @Test
//  @Ignore
  public void testRCV1() throws Exception {
    
    String inputFile = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_ecat_train_500.seq";
    String predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt";
    
    RCV1DatasetInfo.readPredictorNames(predictorNamePath);
    
    runSFOTest(
        RCV1DatasetInfo.get(),
        inputFile,
        37664   // shar
        );
  }
  
  private void runSFOTest(DatasetInfo datasetInfo, String inputFile, int actualBestDim) throws Exception {
    
    // Initialize new empty base model
    SFODriver driver = new SFODriver(
        inputFile,
        TRAIN_OUTPUT_PATH,
        TEST_OUTPUT_PATH,
        REDUCERS_TRAIN,
        REDUCERS_TEST,
        (int)datasetInfo.getNumFeatures());
    
    for (int i=0; i<ITERATIONS; ++i ) {
      
      driver.runSFO();
      
      List<SFODriver.FeatureGain> gains = driver.getGains();
      printTopGains(gains, datasetInfo);
      int bestDim = gains.get(0).getDimension();
      System.out.println("Best dimension: " + bestDim + " (" + datasetInfo.getFeatureName(bestDim) + ")");
//      if (i==0)
//        assertTrue(bestDim == actualBestDim);
      
      driver.addBestFeature();
      
      driver.retrainBaseModel();
    }
  }
  
  private void printTopGains(List<SFODriver.FeatureGain> gains, DatasetInfo datasetInfo) {
    for (int i=0; i<10 && i<gains.size(); ++i) {
      System.out.println("d " + gains.get(i).getDimension() + 
          " (" + datasetInfo.getFeatureName(gains.get(i).getDimension()) 
          + ") gain: " + gains.get(i).getGain());
    }
  }

}
