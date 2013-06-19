package com.andrehacker.ml.preprocess;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.util.CsvReader;
import com.andrehacker.ml.util.MLUtils;
import com.google.common.io.Closeables;

public class CSVToSequenceFile {
  
  public static void transform(String sourcePath, String targetPath, int rows, List<String> predictorNames, String targetName, double targetPositive, double targetNegative) throws Exception {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.getLocal(conf);
    SequenceFile.Writer writer = null;
  
    try {
      CsvReader csv = MLUtils.readData(sourcePath, rows, predictorNames, targetName);
      csv.normalize();
      csv.normalizeClassLabels(targetPositive, targetNegative);
      
      writer = SequenceFile.createWriter(fs, conf, new Path(targetPath),
          IntWritable.class, VectorWritable.class);
      
      IntWritable label = new IntWritable();
      VectorWritable vector = new VectorWritable();
      int i=0;
      for (; i< csv.getData().numRows(); ++i) {
        label.set((int)csv.getY().get(i));
        vector.set(csv.getData().viewRow(i));
        writer.append(label, vector);
      }
      System.out.println("Wrote " + i  + " records into sequence file " + targetPath);
    } finally {
      Closeables.close(writer, true);
    }
  }
}
