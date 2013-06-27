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
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;

import com.andrehacker.ml.datasets.DatasetInfo;
import com.andrehacker.ml.datasets.DonutDatasetInfo;
import com.andrehacker.ml.util.IOUtils;

/**
 * Parallel implementation of Single Feature Optimization algorithm
 * Based on Paper "Parallel Large Scale Feature Selection for Logistic Regression" by Singh et al. 
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.3782
 * 
 * Assumption: Requires total number of features
 */
public class SFOJob extends Configured implements Tool {
  
  private static final String JOB_NAME = "sfo-train";

  static final boolean RUN_LOCAL_MODE = true;
  
  static final double INTERCEPT = 1;    // TODO What to set this to? Train it?
  
  static DatasetInfo modelInfo = DonutDatasetInfo.get();
  
  private String inputFileLocal;
  private String inputFileHdfs;
  private String jarPath;
  private String configFilePath;
  private String outputPath;
  private int reducers;
  
  public SFOJob(String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      String jarPath,
      String configFilePath,
      int reducers) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
    this.jarPath = jarPath;
    this.configFilePath = configFilePath;
    this.reducers = reducers;
  }
  
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
    
    if (RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
      IOUtils.deleteRecursively(outputPath);
      inputFile = inputFileLocal;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = inputFileHdfs;
      conf.addResource(new Path(configFilePath));
      
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", jarPath);
      
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", reducers);
      
      // Delete old output dir
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(outputPath);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());
    
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(SFOIntermediateWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
  
    job.setMapperClass(SFOMapper.class);
//    job.setCombinerClass(Reduce.class);
    job.setReducerClass(SFOReducer.class);
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
//    job.setOutputFormatClass(TextOutputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    
    return job;
  }

}
