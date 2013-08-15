package de.tuberlin.dima.ml.mapred.preprocess;

import java.io.BufferedReader;
import java.io.File;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.inputreader.RCV1VectorReader;
import de.tuberlin.dima.ml.util.MLUtils;

public class RCV1ToSeq {
  
  /**
   * Transforms predefined vector files into sequence files
   * 
   * Assumes that vector and label files are downloaded from this source:
   * 
   * Lewis, D. D.  RCV1-v2/LYRL2004:
   * The LYRL2004 Distribution of the RCV1-v2 Text Categorization Test Collection (14-Oct-2005 Version).
   * http://www.jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm. 
   * 
   * The files have a predefined test/training split which may not be changed!
   * 
   * Stores only the document id and main categories:
   * - CCAT (Corporate/Industrial)
   * - ECAT (Economics)
   * - GCAT (Government/Social)
   * - MCAT (Markets)
   * 
   */
  public static void transform (String folderPath, String positiveClassName, String trainingOutputPath, String testOutputPath, int limit) throws Exception {
    
    if ((new File(trainingOutputPath)).exists() || (new File(testOutputPath)).exists()) {
      throw new Exception("Output file(s) already exists, stop");
    }
    
    List<String> trainingFile = Lists.newArrayList(folderPath + "vectors/lyrl2004_vectors_train.dat");
    String labelPath = folderPath + "rcv1-v2.topics.qrels";
    
    List<String> testFiles = Lists.newArrayList(folderPath + "vectors/lyrl2004_vectors_test_pt0.dat", 
        folderPath + "vectors/lyrl2004_vectors_test_pt1.dat",
        folderPath + "vectors/lyrl2004_vectors_test_pt2.dat",
        folderPath + "vectors/lyrl2004_vectors_test_pt3.dat");
    
    int maxFeatureId = 47237;
    int labelRows = 810935;

    RCV1ToSeq.transform(
        trainingFile, 
        labelPath,
        positiveClassName,
        trainingOutputPath,
        maxFeatureId,
        labelRows,
        limit);
    
    RCV1ToSeq.transform(
        testFiles, 
        labelPath,
        positiveClassName,
        testOutputPath,
        maxFeatureId,
        labelRows,
        limit);
  }
  
  private static void transform (
      List<String> sourcePaths,
      String labelPath,
      String positiveClassName,
      String targetPath,
      int maxFeatureId,
      int labelRows,
      int limit) throws Exception {

    Vector y = RCV1VectorReader.readTarget(labelRows+1, labelPath, positiveClassName);
    
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.getLocal(conf);
    SequenceFile.Writer writer = null;
    BufferedReader reader = null;
    
    int count=0;
    try {
      writer = SequenceFile.createWriter(fs, conf, new Path(targetPath),
          IntWritable.class, VectorWritable.class);
      
      IntWritable label = new IntWritable();
      VectorWritable vector = new VectorWritable();
      
      boolean stop=false;
      for (String sourcePath : sourcePaths) {
        reader = MLUtils.open(sourcePath);
        String line;
        while ((line = reader.readLine()) != null) {
          Vector v = new RandomAccessSparseVector(maxFeatureId);
          int docId = RCV1VectorReader.readVector(v, line);

          vector.set(v);
          label.set((int)y.get(docId));
          
          writer.append(label, vector);
          
          ++count;
          if ((limit != -1) && (count >= limit)) {
            stop=true;
            break;
          }
        }
        Closeables.close(reader, true);
        if (stop) break;
      }
      System.out.println("Wrote " + count  + " records into sequence file " + targetPath);
    } finally {
      Closeables.close(writer, true);
    }
  }
  

  public static void main(String[] args) throws Exception {

    String inputPath = "/home/andre/dev/datasets/RCV1-v2/";
    String outputPath = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/";
    String positiveClass = "ecat"; // Singh uses ECAT
    
    boolean createBig = false;
    boolean createSmall = true;
//    int[] smallSizes = new int[] {5000, 10000, 20000};
    int[] smallSizes = new int[] {100, 500, 1000, 2000};
    
    // ---- Create sequence files with single label ----
    if (createBig) {
      RCV1ToSeq.transform(
          inputPath, 
          positiveClass,
          outputPath + "lyrl2004_vectors_" + positiveClass + "_train.seq", 
          outputPath + "lyrl2004_vectors_" + positiveClass + "_test.seq",
          -1);
  
      // ---- Create sequence files with all labels ----
      RCV1ToSeqMultiLabel.transform(
          inputPath,
          outputPath + "lyrl2004_vectors_train.seq", 
          outputPath + "lyrl2004_vectors_test.seq",
          -1);
      
      printFirstRecords(outputPath + "lyrl2004_vectors_" + positiveClass + "_train.seq");
    }

    if (createSmall) {
      // ---- Produce a smaller version (single and multi label) ----
      for (int size : smallSizes) {
        String smallTrainingOutputPath = outputPath + "lyrl2004_vectors_" + positiveClass + "_train_" + size + ".seq";
        String smallTestOutputPath = outputPath + "lyrl2004_vectors_" + positiveClass + "_test_" + size + ".seq";
        RCV1ToSeq.transform(
            inputPath, 
            positiveClass, 
            smallTrainingOutputPath, 
            smallTestOutputPath, 
            size);
        
        RCV1ToSeqMultiLabel.transform(
            inputPath,
            outputPath + "lyrl2004_vectors_train_" + size + ".seq", 
            outputPath + "lyrl2004_vectors_test_" + size + ".seq",
            size);
      }
    }
  }
  
  private static void printFirstRecords(String sequenceFilePath) {
    
    Configuration conf = new Configuration();
    
    System.out.println("Print first records out of training sequence file:");
    
    int n = 0;
    for (Pair<IntWritable, VectorWritable> labeledRecord : 
        new SequenceFileIterable<IntWritable, VectorWritable>(new Path(sequenceFilePath), conf)) {

      IntWritable label = labeledRecord.getFirst();
      System.out.println("Label: " + label.get());

      Vector features = labeledRecord.getSecond().get();

      System.out.println("- Features: " + features.getNumNondefaultElements() + " of " + features.size());
      System.out.println("- Vec: " + features.toString());
      
      if (++n == 5) {
        break;
      }
    }
  }

}
