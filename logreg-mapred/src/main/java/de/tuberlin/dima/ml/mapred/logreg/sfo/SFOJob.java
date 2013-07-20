package de.tuberlin.dima.ml.mapred.logreg.sfo;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import de.tuberlin.dima.ml.mapred.AbstractHadoopJob;

/**
 * Parallel implementation of Single Feature Optimization algorithm
 * Based on Paper "Parallel Large Scale Feature Selection for Logistic Regression" by Singh et al. 
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.3782
 * 
 * Assumption: Requires total number of features
 */
public class SFOJob extends AbstractHadoopJob {
  
  private static final String JOB_NAME = "sfo-train";
  
  public static enum SFO_COUNTER { 
    TRAIN_OVERFLOWS
  }
  
  private String inputFile;
  private String outputPath;
  
  private int reducers;
  
  public SFOJob(String inputFile,
      String outputPath,
      int reducers) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.reducers = reducers;
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    Job job = prepareJob(
        JOB_NAME, 
        reducers, 
        SFOMapper.class, 
        SFOReducer.class, 
        IntWritable.class,
        SFOIntermediateWritable.class,
        IntWritable.class,
        DoubleWritable.class,
        SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);

    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
