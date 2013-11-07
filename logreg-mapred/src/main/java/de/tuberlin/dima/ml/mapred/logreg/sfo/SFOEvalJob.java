package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.net.URI;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import de.tuberlin.dima.ml.mapred.AbstractHadoopJob;

/**
 * Validates the previously built models
 * Computes the gain regarding a metric (here log-likelihood) for all models
 * compared to the base model.
 * Afterwards we can choose the model with the highest gain.
 * 
 * Some remarks on how we (efficiently) compute the gain:
 * There are many models (as many as remaining features), so it is infeasible 
 * to compute the complete metric (e.g. log-likelihood) for all models.
 * So we only compute the gain regarding log-likelihood compared to the base model.
 * To compute the gain for a model, which differs from base model just in on additional
 * coefficient, we only need to compute the likelihood for the items that actually
 * have this feature. In sparse models (text), this safes a lot of time. 
 */
public class SFOEvalJob extends AbstractHadoopJob {

  private static final String JOB_NAME = "sfo-eval";
  
  public static final String CONF_KEY_IS_MULTILABEL_INPUT = "multilabel-input";
  public static final String CONF_KEY_NUM_FEATURES = "num-features";
  public static final String CONF_KEY_POSITIVE_CLASS = "positive-class";
  public static final String CONF_KEY_TRAIN_OUTPUT = "train-output-path";
  public static final String CONF_KEY_COLLECT_DATASET_STATS = "collect-dataset-stats";
  
  // Counter to compute sparsity (number non-zero values)
  public static enum SFO_EVAL_COUNTER { 
	NUM_NON_ZEROS
  }
  private static final boolean collectDatasetStats = false;
  
  private String inputFile;
  private boolean isMultilabelInput;
  private String outputPath;
  private int numReduceTasks;
  private int numFeatures;
  private int positiveClass;
  private String trainOutputPath;
  private String baseModelPath;
  
  public SFOEvalJob(
      String inputFile,
      boolean isMultilabelInput,
      int positiveClass,
      String outputPath,
      int numReduceTasks,
      int numFeatures,
      String trainOutputPath,
      String baseModelPath) {
    this.inputFile = inputFile;
    this.isMultilabelInput = isMultilabelInput;
    this.positiveClass = positiveClass;
    this.outputPath = outputPath;
    this.numReduceTasks = numReduceTasks;
    this.numFeatures = numFeatures;
    this.trainOutputPath = trainOutputPath;
    this.baseModelPath = baseModelPath;
  }
  
  /**
   * Will be called from ToolRunner internally
   * Hopefully passes us only the args after generic options
   */
  @Override
  public int run(String[] args) throws Exception {
    
    Job job = prepareJob(
        JOB_NAME, 
        numReduceTasks,
        SFOEvalMapper.class, 
        SFOEvalReducer.class, 
        IntWritable.class,
        DoubleWritable.class,
        IntWritable.class,
        DoubleWritable.class,
        TextInputFormat.class,  // SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);
    
    // This might not always be ideal, 
    // e.g. if the number of dimensions is large 
    // and there are only few records per dimension
    job.setCombinerClass(SFOEvalReducer.class);
    
    job.getConfiguration().set(CONF_KEY_IS_MULTILABEL_INPUT, Boolean.toString(isMultilabelInput));
    job.getConfiguration().set(CONF_KEY_NUM_FEATURES, Integer.toString(numFeatures));
    job.getConfiguration().set(CONF_KEY_POSITIVE_CLASS, Integer.toString(positiveClass));
    job.getConfiguration().set(CONF_KEY_TRAIN_OUTPUT, trainOutputPath);
    job.getConfiguration().set(CONF_KEY_COLLECT_DATASET_STATS, Boolean.toString(collectDatasetStats));
    
//    cleanupOutputDirectory(outputPath);

    job.addCacheFile(new URI(baseModelPath));
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
