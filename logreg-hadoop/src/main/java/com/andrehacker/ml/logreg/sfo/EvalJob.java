package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;

import com.andrehacker.ml.util.IOUtils;
import com.andrehacker.ml.writables.DoublePairWritable;

/**
 * Validates the previously built models
 * Computes the gain regarding a metric (here log-likelihood) for all models
 * compared to the base model.
 * Afterwards we can choose the model with the highest gain.
 * Choosing multiple models is probably a bad idea, if there are many correlated 
 * features.
 * 
 * Some remarks, how we (efficiently) compute the gain:
 * There are many models (as many as remaining features), so it is infeasible 
 * to compute the complete metric (e.g. log-likelihood) for all models.
 * So we only compute the gain regarding log-likelihood compared to the base model.
 * To compute the gain for a model, which differs from base model just in on additional
 * coefficient, we only need to compute the likelihood for the items that actually
 * have this feature. In sparse models (text), this safes a lot of time. 
 * 
 * @author Andre Hacker
 * 
 */
public class EvalJob extends Configured implements Tool {

  private static final String JOB_NAME = "sfo-eval";
  
  private String inputFileLocal;
  private String inputFileHdfs;
  private String outputPath;
  private int reducers;
  
  public EvalJob(
      String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      int reducers) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
    this.reducers = reducers;
  }
  
  /**
   * Will be called from ToolRunner internally
   * Hopefully passes us only the args after generic options
   */
  @Override
  public int run(String[] args) throws Exception {
    
    Job job = prepareJob();
    
    return job.waitForCompletion(true) ? 0 : 1;
  }
  
  private Job prepareJob() throws IOException {
    
    System.out.println("-----------------");
    System.out.println("Prepare Job: " + JOB_NAME);
    System.out.println("-----------------");
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());

    String inputFile = "";
    if (GlobalJobSettings.RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
      IOUtils.deleteRecursively(outputPath);
      inputFile = inputFileLocal;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = inputFileHdfs;
      conf.addResource(new Path(GlobalJobSettings.CONFIG_FILE_PATH));
      
      conf.set("mapred.jar", GlobalJobSettings.JAR_PATH);
      
      conf.setInt("mapred.reduce.tasks", reducers);
      
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(outputPath);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(DoublePairWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
  
    job.setMapperClass(EvalMapper.class);
    job.setReducerClass(EvalReducer.class);
    
//    job.setInputFormatClass(KeyValueTextInputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
//    job.setOutputFormatClass(TextOutputFormat.class);
  
    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    
    return job;
  }

}
