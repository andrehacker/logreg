package com.andrehacker.ml.preprocess;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

import com.google.common.collect.Lists;

public class CSVToSequenceFileTest {
  
  @Test
  public void test() throws Exception {
    List<String> predictorNames = Lists.newArrayList(new String[] {
//      "x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c", "bias"   // k, k0 not in testfile!
//      "x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c"
        "x", "y", "shape", "color", "xx", "xy", "yy", "a", "b", "c"
//      "x", "y", "shape", "xx", "xy", "yy", "a", "b", "c"    // nice chart
//      "x", "y", "shape", "a", "b", "c"
//      "x", "y", "a", "b", "c"    // like Mahout page 252
//      "x", "c"    // Adding a or b is REALLY BAD. However x and y are good. Shape is okay.
//        "y", "x"
     });
    
    String testFile = "donut-test.csv";
    String trainingFile = "donut.csv";
    int rows = 40;
    String targetName = "color";
    double targetPositive = 2d;
    double targetNegative = 1d;
    
    CSVToSequenceFile.transform(
        testFile, 
        testFile.concat(".seq"), 
        rows, 
        predictorNames, 
        targetName, 
        targetPositive, 
        targetNegative);
    
    CSVToSequenceFile.transform(
        trainingFile, 
        trainingFile.concat(".seq"), 
        rows, 
        predictorNames, 
        targetName, 
        targetPositive, 
        targetNegative);
    
    Configuration conf = new Configuration();

    System.out.println("Print first records out of sequence file:");
    
    int n = 0;
    for (Pair<IntWritable, VectorWritable> labeledRecord : 
        new SequenceFileIterable<IntWritable, VectorWritable>(new Path(testFile.concat(".seq")), conf)) {

      System.out.println("Label: " + labeledRecord.getFirst().get());

      Vector features = labeledRecord.getSecond().get();

      System.out.println("- Features: " + features.getNumNondefaultElements() + " of " + features.size());
      System.out.println("- Vec: " + features.toString());

      if (++n == 2) {
        break;
      }
    }
  }

}
