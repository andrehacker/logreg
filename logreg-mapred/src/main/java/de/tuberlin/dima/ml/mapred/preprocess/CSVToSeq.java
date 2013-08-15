package de.tuberlin.dima.ml.mapred.preprocess;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.util.CsvReader;
import de.tuberlin.dima.ml.util.MLUtils;

public class CSVToSeq {
  
  /**
   * Stores result also as text file (csv), relevant if data were normalized
   */
  public static void transform(
      String sourcePath,
      String targetPath,
      String targetCsvPath,
      int rows, 
      List<String> predictorNames, 
      String targetName,
      boolean normalizeData,
      boolean normalizeLabels,
      double targetPositive, 
      double targetNegative) throws Exception {
    
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.getLocal(conf);
    SequenceFile.Writer writer = null;
    Writer csvWriter = new BufferedWriter(new OutputStreamWriter(
        new FileOutputStream(targetCsvPath), "UTF-8"));
  
    try {
      CsvReader csv = MLUtils.readData(sourcePath, rows, predictorNames, targetName, false);
      if (normalizeData){
        csv.normalize();
      }
      if (normalizeLabels) {
        csv.normalizeClassLabels(targetPositive, targetNegative);
      }
      
      writer = SequenceFile.createWriter(fs, conf, new Path(targetPath),
          IntWritable.class, VectorWritable.class);
      
      IntWritable label = new IntWritable();
      VectorWritable vector = new VectorWritable();
      int i=0;
      for (; i< csv.getData().numRows(); ++i) {
        label.set((int)csv.getY().get(i));
        vector.set(csv.getData().viewRow(i));
        writer.append(label, vector);
        
        for (int col=0; col<vector.get().size(); ++col) {
          if (col>0) csvWriter.write(",");
          csvWriter.write(Double.toString(vector.get().get(col)));
        }
        csvWriter.write("\n");
      }
      System.out.println("Wrote " + i  + " records into sequence file " + targetPath);
    } finally {
      Closeables.close(writer, true);
      Closeables.close(csvWriter, true);
    }
  }
  
  public static void main(String[] args) throws Exception {

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
