package de.tuberlin.dima.ml.preprocess;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

import de.tuberlin.dima.ml.writables.IDAndLabels;

public class RCV1ToSequenceFileTest {

  @Test
  public void testTransform() throws Exception {
    
    String folderPath = "/home/andre/dev/datasets/RCV1-v2/";
    String trainingOutputPath = folderPath + "vectors/lyrl2004_vectors_train.seq";
    String testOutputPath = folderPath + "vectors/lyrl2004_vectors_test.seq";
    
    RCV1ToSeqMultiLabel.transform(folderPath, trainingOutputPath, testOutputPath, -1);

    // Produce a smaller version
    int limit = 5000;
    String smallTrainingOutputPath = folderPath + "vectors/lyrl2004_vectors_train_" + limit + ".seq";
    String smallTestOutputPath = folderPath + "vectors/lyrl2004_vectors_test_" + limit + ".seq";
    RCV1ToSeqMultiLabel.transform(folderPath, smallTrainingOutputPath, smallTestOutputPath, limit);
    
    Configuration conf = new Configuration();
    
    System.out.println("Print first records out of training sequence file:");
    
    int n = 0;
    for (Pair<IDAndLabels, VectorWritable> labeledRecord : 
        new SequenceFileIterable<IDAndLabels, VectorWritable>(new Path(trainingOutputPath), conf)) {

      IDAndLabels idAndLabels = labeledRecord.getFirst();
      System.out.println("Label:" + idAndLabels.getId() + " Label (CCAT): " + idAndLabels.getLabels().get(0));

      Vector features = labeledRecord.getSecond().get();

      System.out.println("- Features: " + features.getNumNondefaultElements() + " of " + features.size());
      System.out.println("- Vec: " + features.toString());

      if (++n == 5) {
        break;
      }
    }
    
  }
  
  


}
