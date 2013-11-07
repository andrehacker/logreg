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
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;

public class SFOToolsHadoop {
  
  /**
   * Write the current base model in the sequencefile format.
   * Can be written to either hdfs or local file
   * The key will be the base model, the value will be null.
   * This file will be broadcasted via Distributed Cache afterwards.
   * 
   * @param baseModel base model instance
   * @param conf The filesystem to store the file will be derived from this
   * @param baseModelPath The absolute path where to store the file in the file system
   */
  static void writeBaseModel(IncrementalModel baseModel, Configuration conf, String baseModelPath) throws IOException {
    
    FileSystem fs = FileSystem.get(conf);
    baseModelPath = fs.getUri() + baseModelPath;
    System.out.println("Write basemodel to " + baseModelPath);
    SequenceFile.Writer writer = null;
    try {
      writer = SequenceFile.createWriter(
    	  conf,
          Writer.file(new Path(baseModelPath)),
          Writer.keyClass(IncrementalModelWritable.class),
          Writer.valueClass(NullWritable.class));
      writer.append(new IncrementalModelWritable(baseModel), NullWritable.get());
    } finally {
      Closeables.close(writer, true);
    }
  }

  /**
   * Read base model from distributed cache (stored in sequence file format)
   * 
   * @param context context from the UDF
   * @return base model instance
   */
  static IncrementalModel readBaseModelFromDC(
      @SuppressWarnings("rawtypes") Context context) throws IOException {
	// I am not aware of any non-deprecated method to do this;-(
    @SuppressWarnings("deprecation")
    Path[] cachedFiles = context.getLocalCacheFiles();
    Path localPath = new Path("file://" + cachedFiles[0].toString());
    IncrementalModel baseModel = null;
    for (Pair<IncrementalModelWritable, NullWritable> all : new SequenceFileIterable<IncrementalModelWritable, NullWritable>(
        localPath, context.getConfiguration())) {
      baseModel = all.getFirst().getModel();
      System.out.println("Read base model from " + localPath);
    }
    return baseModel;
  }
  
  /**
   * Read resulting feature gains from evaluation job.
   * 
   * @param path job output path
   * @param conf used to derive the filesystem for the output path
   * @return list of gains for each newly trained approximate model
   */
  static List<FeatureGain> readEvalResult(String path, Configuration conf) throws IOException {
    
    List<FeatureGain> list = Lists.newArrayList();
    
    Path dir = new Path(path);
    FileSystem fs = FileSystem.get(conf);
    FileStatus[] statusList = fs.listStatus(dir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        if (path.getName().startsWith("part-r")) return true;
        else return false;
      }
    });
    for (FileStatus status : statusList) {

      SequenceFile.Reader reader = new SequenceFile.Reader(conf, Reader.file(status.getPath()));
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
  
  /**
   * Read trained coefficients from the output of the training job
   * 
   * @param conf filesystem will be derived from this configuration
   * @param highestFeatureId highest index of all features. This usually equals the number of features
   * @param trainOutputPath output path 
   * @return
   */
  static List<Double> readTrainedCoefficients(Configuration conf, String trainOutputPath, int highestFeatureId) throws IOException {
    
    // Read trained coefficients into map: dimension -> coefficient
    List<Double> coefficients = Arrays.asList(new Double[highestFeatureId]);
    
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
      SequenceFile.Reader reader = new SequenceFile.Reader(conf, Reader.file(status.getPath()));
      try {
        IntWritable dimension = new IntWritable();
        DoubleWritable coefficient = new DoubleWritable();
        while (reader.next(dimension, coefficient)) {
          coefficients.set(dimension.get(), coefficient.get());
        }
      } finally {
        Closeables.close(reader, true);
      }
    }
    return coefficients;
  }

}
