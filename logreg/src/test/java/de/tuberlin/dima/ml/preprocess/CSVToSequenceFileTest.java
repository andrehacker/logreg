package de.tuberlin.dima.ml.preprocess;

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

import de.tuberlin.dima.ml.preprocess.CSVToSeq;

public class CSVToSequenceFileTest {
  
  @Test
  public void test() throws Exception {
    
    List<String> predictorNames = Lists.newArrayList(new String[] {
        "x", "y", "shape", "xx", "xy", "yy", "a", "b", "c"     // All relevant fields which are in both, training and test file
     });
    
    String testFile = "donut-test.csv";
    String trainingFile = "donut.csv";
    int rows = 40;
    String targetName = "color";
    double targetPositive = 2d;
    double targetNegative = 1d;
    
    CSVToSeq.transform(
        testFile, 
        testFile.concat(".seq"), 
        testFile.concat(".normalized"), 
        rows, 
        predictorNames,
        targetName, 
        true,
        true,
        targetPositive, 
        targetNegative);
    
    CSVToSeq.transform(
        trainingFile,
        trainingFile.concat(".seq"),
        trainingFile.concat(".normalized"),
        rows,
        predictorNames,
        targetName,
        true,
        true,
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
