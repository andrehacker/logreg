package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;

public class SFOToolsHadoop {
  
  private static final String BASE_MODEL_PATH = "sfo-base-model.seq";
  
  /**
   * Makes the current base model available to mappers/reducers via hdfs
   * 
   * We use hdfs (and not distributed cache) because the base model remains
   * the same for train and test job.
   * 
   * TODO Improvement: Appending only every single added dimension would be nice
   * to reduce startup costs
   */
  static void writeBaseModel(IncrementalModel baseModel, Configuration conf) throws IOException {
    
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer writer = null;
    String baseModelPath = fs.getUri() + "/" + BASE_MODEL_PATH;
    System.out.println("Write basemodel to " + baseModelPath);
    try {
      writer = SequenceFile.createWriter(fs, conf, new Path(baseModelPath),
          IncrementalModelWritable.class, NullWritable.class);
      writer.append(new IncrementalModelWritable(baseModel), NullWritable.get());
    } finally {
      Closeables.close(writer, true);
    }
  }
    
  static IncrementalModel readBaseModel(Configuration conf) throws IOException {
    System.out.println("READ BASE MODEL FS.DEFAULTFS: " + conf.get("fs.defaultFS"));
    IncrementalModel baseModel = null;
    FileSystem fs = FileSystem.get(conf);
//    String baseModelPath = conf.get("fs.defaultFS") + "/" + BASE_MODEL_PATH;
    String baseModelPath = fs.getUri() + "/" + BASE_MODEL_PATH;
    System.out.println("Read basemodel from " + baseModelPath);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(baseModelPath), conf);
    try {
      IncrementalModelWritable baseModelWritable = new IncrementalModelWritable();
      // model is stored in key
      reader.next(baseModelWritable);
      baseModel = baseModelWritable.getModel();
    } finally {
      Closeables.close(reader, true);
    }
    
    return baseModel;
  }
  
  static List<FeatureGain> readEvalResult(String path, Configuration conf) throws IOException {
    
    List<FeatureGain> list = Lists.newArrayList();
    
//    Configuration conf = HadoopUtils.createConfiguration(hdfsAddress);
    
    Path dir = new Path(path);
    FileSystem fs = FileSystem.get(conf);
    FileStatus[] statusList = fs.listStatus(dir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        if (path.getName().startsWith("part-r")) return true;
        else return false;
      }
    });
//    System.out.println("Read gain from " + statusList.length + " files");
    for (FileStatus status : statusList) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
      try {
        IntWritable dimension = new IntWritable();
        DoubleWritable gain = new DoubleWritable();
        while (reader.next(dimension, gain)) {
          list.add(new FeatureGain(dimension.get(), gain.get()));
        }
      } finally {
        Closeables.close(reader, true);
      }
    }
    
    return list;
  }
  
  static List<Double> readTrainedCoefficients(Configuration conf, int numFeatures, String trainOutputPath) throws IOException {
    
    // Read trained coefficients into map: dimension -> coefficient
    List<Double> coefficients = Arrays.asList(new Double[numFeatures]);
    
    Path dir = new Path(trainOutputPath);
    FileSystem fs = FileSystem.get(conf);
    FileStatus[] statusList = fs.listStatus(dir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        if (path.getName().startsWith("part-r")) return true;
        else return false;
      }
    });
//    System.out.println("Read trained coefficients from " + statusList.length + " files");
    for (FileStatus status : statusList) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
      try {
        IntWritable dimension = new IntWritable();
        DoubleWritable coefficient = new DoubleWritable();
        while (reader.next(dimension, coefficient)) {
          coefficients.set(dimension.get(), coefficient.get());
//          System.out.println(dimension.get() + ": " + coefficient.get());
        }
      } finally {
        Closeables.close(reader, true);
      }
    }
    return coefficients;
  }

}
