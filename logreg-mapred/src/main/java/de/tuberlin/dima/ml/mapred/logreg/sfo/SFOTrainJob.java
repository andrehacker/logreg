package de.tuberlin.dima.ml.mapred.logreg.sfo;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import de.tuberlin.dima.ml.mapred.AbstractHadoopJob;

/**
 * Parallel implementation of Single Feature Optimization algorithm
 * Based on Paper "Parallel Large Scale Feature Selection for Logistic Regression" by Singh et al. 
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.3782
 */
public class SFOTrainJob extends AbstractHadoopJob {
  
  private static final String JOB_NAME = "sfo-train";

  static final String CONF_KEY_NUM_FEATURES = "num-features";
  static final String CONF_KEY_LABEL_INDEX = "label-index";
  
  public static enum SFO_COUNTER { 
    TRAIN_OVERFLOWS
  }
  
  private String inputFile;
  private String outputPath;
  
  private int reducers;
  
  private int numFeatures;
  private int labelIndex;
  
  public SFOTrainJob(String inputFile,
      String outputPath,
      int reducers,
      int numFeatures,
      int labelIndex) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.reducers = reducers;
    this.numFeatures = numFeatures;
    this.labelIndex = labelIndex;
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    // TODO To really implement Tool we need to parse the arguments!
    // Currently the client has to call the constructor before
    
    Job job = prepareJob(
        JOB_NAME, 
        reducers, 
        SFOTrainMapper.class, 
        SFOTrainReducer.class, 
        IntWritable.class,
        SFOIntermediateWritable.class,
        IntWritable.class,
        DoubleWritable.class,
        TextInputFormat.class, // SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);
    
    job.getConfiguration().set(CONF_KEY_NUM_FEATURES, Integer.toString(numFeatures));
    job.getConfiguration().set(CONF_KEY_LABEL_INDEX, Integer.toString(labelIndex));

    //    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
