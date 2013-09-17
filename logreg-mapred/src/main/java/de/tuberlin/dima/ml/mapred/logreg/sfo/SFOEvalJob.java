package de.tuberlin.dima.ml.mapred.logreg.sfo;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import de.tuberlin.dima.ml.mapred.AbstractHadoopJob;
import de.tuberlin.dima.ml.mapred.writables.DoublePairWritable;

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
  
  private String inputFile;
  private boolean isMultilabelInput;
  private String outputPath;
  private int numReduceTasks;
  private int numFeatures;
  private int positiveClass;
  private String trainOutputPath;
  
  public SFOEvalJob(
      String inputFile,
      boolean isMultilabelInput,
      int positiveClass,
      String outputPath,
      int numReduceTasks,
      int numFeatures,
      String trainOutputPath) {
    this.inputFile = inputFile;
    this.isMultilabelInput = isMultilabelInput;
    this.positiveClass = positiveClass;
    this.outputPath = outputPath;
    this.numReduceTasks = numReduceTasks;
    this.numFeatures = numFeatures;
    this.trainOutputPath = trainOutputPath;
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
        DoublePairWritable.class,
        IntWritable.class,
        DoubleWritable.class,
        TextInputFormat.class,  // SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);

    job.getConfiguration().set(CONF_KEY_IS_MULTILABEL_INPUT, Boolean.toString(isMultilabelInput));
    job.getConfiguration().set(CONF_KEY_NUM_FEATURES, Integer.toString(numFeatures));
    job.getConfiguration().set(CONF_KEY_POSITIVE_CLASS, Integer.toString(positiveClass));
    job.getConfiguration().set(CONF_KEY_TRAIN_OUTPUT, trainOutputPath);
    
//    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
