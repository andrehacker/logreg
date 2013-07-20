package de.tuberlin.dima.ml.mapred.preprocess;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public class RCV1ToSeqTest {
  
  private static final String INPUT_PATH = "/home/andre/dev/datasets/RCV1-v2/";
  private static final String OUTPUT_PATH = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/";
  
  private static final String POSITIVE_CLASS = "ecat"; // Singh uses ECAT

  @Test
  public void testTransform() throws Exception {
    
    boolean createBig = false;
    boolean createSmall = true;
//    int[] smallSizes = new int[] {5000, 10000, 20000};
    int[] smallSizes = new int[] {100, 500, 1000, 2000};
    
    // ---- Create sequence files with single label ----
    if (createBig) {
      RCV1ToSeq.transform(
          INPUT_PATH, 
          POSITIVE_CLASS,
          OUTPUT_PATH + "lyrl2004_vectors_" + POSITIVE_CLASS + "_train.seq", 
          OUTPUT_PATH + "lyrl2004_vectors_" + POSITIVE_CLASS + "_test.seq",
          -1);
  
      // ---- Create sequence files with all labels ----
      RCV1ToSeqMultiLabel.transform(
          INPUT_PATH,
          OUTPUT_PATH + "lyrl2004_vectors_train.seq", 
          OUTPUT_PATH + "lyrl2004_vectors_test.seq",
          -1);
      
      printFirstRecords(OUTPUT_PATH + "lyrl2004_vectors_" + POSITIVE_CLASS + "_train.seq");
    }

    if (createSmall) {
      // ---- Produce a smaller version (single and multi label) ----
      for (int size : smallSizes) {
        String smallTrainingOutputPath = OUTPUT_PATH + "lyrl2004_vectors_" + POSITIVE_CLASS + "_train_" + size + ".seq";
        String smallTestOutputPath = OUTPUT_PATH + "lyrl2004_vectors_" + POSITIVE_CLASS + "_test_" + size + ".seq";
        RCV1ToSeq.transform(
            INPUT_PATH, 
            POSITIVE_CLASS, 
            smallTrainingOutputPath, 
            smallTestOutputPath, 
            size);
        
        RCV1ToSeqMultiLabel.transform(
            INPUT_PATH,
            OUTPUT_PATH + "lyrl2004_vectors_train_" + size + ".seq", 
            OUTPUT_PATH + "lyrl2004_vectors_test_" + size + ".seq",
            size);
      }
    }
  }
  
  private void printFirstRecords(String sequenceFilePath) {
    
    Configuration conf = new Configuration();
    
    System.out.println("Print first records out of training sequence file:");
    
    int n = 0;
    for (Pair<IntWritable, VectorWritable> labeledRecord : 
        new SequenceFileIterable<IntWritable, VectorWritable>(new Path(sequenceFilePath), conf)) {

      IntWritable label = labeledRecord.getFirst();
      System.out.println("Label (" + POSITIVE_CLASS + "): " + label.get());

      Vector features = labeledRecord.getSecond().get();

      System.out.println("- Features: " + features.getNumNondefaultElements() + " of " + features.size());
      System.out.println("- Vec: " + features.toString());
      
      if (++n == 5) {
        break;
      }
    }
  }


}
