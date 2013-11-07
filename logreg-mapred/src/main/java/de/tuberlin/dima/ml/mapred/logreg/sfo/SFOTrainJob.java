package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.net.URI;

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

  public static final String CONF_KEY_IS_MULTILABEL_INPUT = "multilabel-input";
  public static final String CONF_KEY_POSITIVE_CLASS = "positive-class";
  public static final String CONF_KEY_NUM_FEATURES = "num-features";
  public static final String CONF_KEY_NEWTON_MAX_ITERATIONS = "newton-max-iterations";
  public static final String CONF_KEY_NEWTON_TOLERANCE = "newton-tolerance";
  public static final String CONF_KEY_REGULARIZATION = "regularization";
  public static final String CONF_KEY_COLLECT_DATASET_STATS = "collect-dataset-stats";
  
  // Counter to compute sparsity (number non-zero values)
  public static enum SFO_TRAIN_COUNTER { 
	NUM_NON_ZEROS
  }
  private static final boolean collectDatasetStats = false;
  
  private String inputFile;
  private boolean isMultilabelInput;
  private int positiveClass;
  private String outputPath;
  private int numReduceTasks;
  private int numFeatures;
  private double newtonTolerance;
  private int newtonMaxIterations;
  private double regularization;
  private String baseModelPath;
  
  public SFOTrainJob(
      String inputFile,
      boolean isMultilabelInput,
      int positiveClass,
      String outputPath,
      int numFeatures,
      double newtonTolerance,
      int newtonMaxIterations,
      double regularization,
      int numReduceTasks,
      String baseModelPath
      ) {
    this.inputFile = inputFile;
    this.isMultilabelInput = isMultilabelInput;
    this.positiveClass = positiveClass;
    this.outputPath = outputPath;
    this.numFeatures = numFeatures;
    this.newtonTolerance = newtonTolerance;
    this.newtonMaxIterations = newtonMaxIterations;
    this.regularization = regularization;
    this.numReduceTasks = numReduceTasks;
    this.baseModelPath = baseModelPath;
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    // TODO To really implement Tool we need to parse the arguments!
    // Currently the client has to call the constructor before
    
    Job job = prepareJob(
        JOB_NAME, 
        numReduceTasks, 
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
    job.getConfiguration().set(CONF_KEY_IS_MULTILABEL_INPUT, Boolean.toString(isMultilabelInput));
    job.getConfiguration().set(CONF_KEY_POSITIVE_CLASS, Integer.toString(positiveClass));
    job.getConfiguration().set(CONF_KEY_NUM_FEATURES, Integer.toString(numFeatures));
    job.getConfiguration().set(CONF_KEY_NEWTON_TOLERANCE, Double.toString(newtonTolerance));
    job.getConfiguration().set(CONF_KEY_NEWTON_MAX_ITERATIONS, Integer.toString(newtonMaxIterations));
    job.getConfiguration().set(CONF_KEY_REGULARIZATION, Double.toString(regularization));
    job.getConfiguration().set(CONF_KEY_COLLECT_DATASET_STATS, Boolean.toString(collectDatasetStats));

    job.addCacheFile(new URI(baseModelPath));

    //    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
