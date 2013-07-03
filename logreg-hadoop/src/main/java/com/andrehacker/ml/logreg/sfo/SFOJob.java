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
  
  private String inputFileLocal;
  private String inputFileHdfs;
  private String outputPath;
  
  private int reducers;
  
  public SFOJob(String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      int reducers) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
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

    if (GlobalJobSettings.RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
    }
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    
    // ---------- Set Configuration options -----------

    conf.addResource(new Path(GlobalJobSettings.CONFIG_FILE_PATH));
    
    if (!GlobalJobSettings.RUN_LOCAL_MODE) {
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", GlobalJobSettings.JAR_PATH);
      
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", reducers);
    }
    
    // ---------- Cleanup output directory ------------

    if (GlobalJobSettings.RUN_LOCAL_MODE) {
      IOUtils.deleteRecursively(outputPath);
    } else {
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(outputPath);
      hdfs.delete(path, true);
    }
    
    // ---------- Set Job options ---------------------
    
    job.setJarByClass(getClass());
    
    String inputFile = "";
    if (GlobalJobSettings.RUN_LOCAL_MODE) {
      inputFile = inputFileLocal;
    } else {
      inputFile = inputFileHdfs;
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
