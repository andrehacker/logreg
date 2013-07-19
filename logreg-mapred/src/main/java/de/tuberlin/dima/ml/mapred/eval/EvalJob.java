package de.tuberlin.dima.ml.mapred.eval;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.common.IntPairWritable;

import de.tuberlin.dima.ml.mapred.AbstractHadoopJob;

public class EvalJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-validation";
  
  static final String CONF_KEY_TRAIN_OUTPUT = "train-output";
  static final String CONF_KEY_LABEL_DIMENSION = "label-dimension";

  private static final int REDUCE_TASKS = 1;
  
  private String inputFile;
  private String outputPath;
  private String trainOuputPath;
  private int labelDimension;
  
  public EvalJob(String inputFile,
      String outputPath,
      String trainOuputPath,
      int labelDimension) {
    
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.trainOuputPath = trainOuputPath;
    this.labelDimension = labelDimension;
  }

  public int run(String[] args) throws Exception {
    
    Job job = prepareJob(
        JOB_NAME, 
        REDUCE_TASKS, 
        EvalMapper.class, 
        EvalReducer.class, 
        Text.class,
        IntPairWritable.class,
        Text.class,
        Text.class,
        SequenceFileInputFormat.class,
        TextOutputFormat.class,
        inputFile,
        outputPath);
    
    job.getConfiguration().set(CONF_KEY_LABEL_DIMENSION, Integer.toString(labelDimension));
    
    cleanupOutputDirectory(outputPath);
    
    job.getConfiguration().set(CONF_KEY_TRAIN_OUTPUT, trainOuputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
