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

import com.andrehacker.ml.DoublePairWritable;
import com.andrehacker.ml.util.IOUtils;

/**
 * Parallel implementation of Single Feature Optimization algorithm
 * Based on Paper "Parallel Large Scale Feature Selection for Logistic Regression" by Singh et al. 
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.3782
 * 
 * Assumption: Requires total number of features
 */
public class EvalJob extends Configured implements Tool {

  static final int REDUCE_TASKS = 4;
  
  private static final String JOB_NAME = "SFO-EVAL";
  
  private static final String OUTPUT_PATH = "output-sfo-eval";
  
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
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());

    String inputFile = "";
    if (SFOJob.RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
      IOUtils.deleteRecursively(OUTPUT_PATH);
      inputFile = SFOJob.INPUT_FILE_LOCAL;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = SFOJob.INPUT_FILE_HDFS;
      conf.addResource(new Path(SFOJob.CONFIG_FILE_PATH));
      
      conf.set("mapred.jar", SFOJob.JAR_PATH);
      
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
      
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(OUTPUT_PATH);
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
    FileOutputFormat.setOutputPath(job, new Path(OUTPUT_PATH));
    
    return job;
  }

}
