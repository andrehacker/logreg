package de.tuberlin.dima.ml.preprocess;

import java.io.BufferedReader;
import java.io.File;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.inputreader.RCV1VectorReader;
import de.tuberlin.dima.ml.util.MLUtils;
import de.tuberlin.dima.ml.writables.IDAndLabels;

public class RCV1ToSeqMultiLabel {
  
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
  public static void transform (String folderPath, String trainingOutputPath, String testOutputPath, int limit) throws Exception {
    
    if ((new File(trainingOutputPath)).exists() || (new File(testOutputPath)).exists()) {
      throw new Exception("Output file(s) already exists, stop");
    }
    
    List<String> trainingFile = Lists.newArrayList(folderPath + "vectors/lyrl2004_vectors_train.dat");
    String labelPath = folderPath + "rcv1-v2.topics.qrels";
    
    List<String> testFiles = Lists.newArrayList(folderPath + "vectors/lyrl2004_vectors_test_pt0.dat", 
        folderPath + "vectors/lyrl2004_vectors_test_pt1.dat",
        folderPath + "vectors/lyrl2004_vectors_test_pt2.dat",
        folderPath + "vectors/lyrl2004_vectors_test_pt3.dat");
    
    int maxFeatureId = (int)RCV1DatasetInfo.get().getNumFeatures();
    int labelRows = (int)RCV1DatasetInfo.get().getTotal();

    RCV1ToSeqMultiLabel.transform(trainingFile, 
        labelPath,
        trainingOutputPath,
        maxFeatureId,
        labelRows,
        limit);
    
    RCV1ToSeqMultiLabel.transform(testFiles, 
        labelPath,
        testOutputPath,
        maxFeatureId,
        labelRows,
        limit);
  }
  
  private static void transform (List<String> sourcePaths,
      String labelPath,
      String targetPath,
      int maxFeatureId,
      int labelRows,
      int limit) throws Exception {

    DenseVector yC = new DenseVector(labelRows+1);
    DenseVector yE = new DenseVector(labelRows+1);
    DenseVector yG = new DenseVector(labelRows+1);
    DenseVector yM = new DenseVector(labelRows+1);
    RCV1VectorReader.readLabels(labelPath, yC, yE, yG, yM);
    
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.getLocal(conf);
    SequenceFile.Writer writer = null;
    BufferedReader reader = null;
    
    int count=0;
    try {
      writer = SequenceFile.createWriter(fs, conf, new Path(targetPath),
          IDAndLabels.class, VectorWritable.class);
      
      IDAndLabels idAndLabels = new IDAndLabels();
      VectorWritable vector = new VectorWritable();
      
      boolean stop=false;
      for (String sourcePath : sourcePaths) {
        reader = MLUtils.open(sourcePath);
        String line;
        while ((line = reader.readLine()) != null) {
          Vector v = new RandomAccessSparseVector(maxFeatureId);
          int docId = RCV1VectorReader.readVector(v, line);

          vector.set(v);
          idAndLabels.set(docId,
              new DenseVector(
                  new double[] {
                      (int)yC.get(docId),
                      (int)yE.get(docId),
                      (int)yG.get(docId),
                      (int)yM.get(docId)}));
          
          writer.append(idAndLabels, vector);
          
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

}

