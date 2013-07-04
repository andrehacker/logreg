package com.andrehacker.ml.preprocess;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public class RCV1ToSequenceFileTest {

  @Test
  public void testTransform() throws Exception {
    
    String folderPath = "/home/andre/dev/datasets/RCV1-v2/";
    String trainingOutputPath = folderPath + "vectors/lyrl2004_vectors_sfo_train.seq";
    String testOutputPath = folderPath + "vectors/lyrl2004_vectors_sfo_test.seq";
    
    String positiveClassName = "ECAT"; // Singh uses ECAT
    
    RCV1ToSequenceFile.transform(folderPath, positiveClassName, trainingOutputPath, testOutputPath, -1);

    // Produce a smaller version
    int limit = 10000;
    String smallTrainingOutputPath = folderPath + "vectors/lyrl2004_vectors_sfo_train_" + limit + ".seq";
    String smallTestOutputPath = folderPath + "vectors/lyrl2004_vectors_sfo_test_" + limit + ".seq";
    RCV1ToSequenceFile.transform(folderPath, positiveClassName, smallTrainingOutputPath, smallTestOutputPath, limit);
    
    Configuration conf = new Configuration();
    
    System.out.println("Print first records out of training sequence file:");
    
    int n = 0;
    for (Pair<IntWritable, VectorWritable> labeledRecord : 
        new SequenceFileIterable<IntWritable, VectorWritable>(new Path(smallTrainingOutputPath), conf)) {

      IntWritable label = labeledRecord.getFirst();
      System.out.println("Label (" + positiveClassName + "): " + label.get());

      Vector features = labeledRecord.getSecond().get();

      System.out.println("- Features: " + features.getNumNondefaultElements() + " of " + features.size());
      System.out.println("- Vec: " + features.toString());
      
      if (++n == 5) {
        break;
      }
    }
    
  }


}
