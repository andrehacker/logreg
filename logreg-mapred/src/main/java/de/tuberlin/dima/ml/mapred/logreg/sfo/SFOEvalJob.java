package de.tuberlin.dima.ml.mapred.logreg.sfo;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
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
  
  static final String CONF_KEY_NUM_FEATURES = "num-features";
  static final String CONF_KEY_TRAIN_OUTPUT = "train-output-path";
  
  private String inputFile;
  private String outputPath;
  private int reducers;
  private int numFeatures;
  private String trainOutputPath;
  
  public SFOEvalJob(
      String inputFile,
      String outputPath,
      int reducers,
      int numFeatures,
      String trainOutputPath) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.reducers = reducers;
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
        reducers, 
        SFOEvalMapper.class, 
        SFOEvalReducer.class, 
        IntWritable.class,
        DoublePairWritable.class,
        IntWritable.class,
        DoubleWritable.class,
        SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);

    job.getConfiguration().set(CONF_KEY_NUM_FEATURES, Integer.toString(numFeatures));
    job.getConfiguration().set(CONF_KEY_TRAIN_OUTPUT, trainOutputPath);
    
    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
