package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;

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

import com.andrehacker.ml.DatasetInfo;
import com.andrehacker.ml.util.IOUtils;
import com.google.common.collect.Lists;

/**
 * Parallel implementation of Single Feature Optimization algorithm
 * Based on Paper "Parallel Large Scale Feature Selection for Logistic Regression" by Singh et al. 
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.3782
 * 
 * Assumption: Requires total number of features
 */
public class SFOJob extends Configured implements Tool {

  static final boolean RUN_LOCAL_MODE = true;
  
//  static final int FEATURES = 47237;
  static final double INTERCEPT = 1;    // TODO What to set this to? Train it?
  
  // Stats for RCV1-v2:
  // - 47236 is highest term id
  // - 381327 points labeled with CCAT (RCV1-v2)
  // - 810935 is highest document-id
  // - Used CCAT during extraction
  // - Singh used ECAT, giving 3449 positive training instances
  
  static final int REDUCE_TASKS = 4;
  
  private static List<String> predictorNames = Lists.newArrayList(new String[] {
      "x", "y", "shape", "xx", "xy", "yy", "a", "b", "c"
   });
  static DatasetInfo modelInfo = new DatasetInfo(predictorNames);
  
//  private static final String LABEL_FILE_LOCAL = "/home/andre/dev/datasets/RCV1-v2/rcv1-v2.topics_ccat.qrels";
//  private static final String LABEL_FILE_HDFS = "rcv1-v2/rcv1-v2.topics_ccat.qrels";
//  private static final String INPUT_FILE_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train.dat";
//  private static final String INPUT_FILE_HDFS = "rcv1-v2/lyrl2004_vectors_train.dat";
  static final String INPUT_FILE_LOCAL = "/home/andre/dev/datasets/donut/donut.csv.seq";
  static final String INPUT_FILE_HDFS = "donut/donut.csv.seq";
  
  static final String TRAIN_OUTPUT_PATH = "output-sfo-train";
  
  static final String JAR_PATH = "target/logreg-0.0.1-SNAPSHOT-job.jar";
  static final String CONFIG_FILE_PATH = "core-site.xml";

  private static final String JOB_NAME = "SFO-Train";
  
  /**
   * Will be called from ToolRunner internally
   * Hopefully passes us only the args after generic options
   */
  @Override
  public int run(String[] args) throws Exception {
//    if (args.length != 2) {
//      System.err.printf("Usage: %s [generic options] <input> <output>", getClass().getSimpleName());
//      ToolRunner.printGenericCommandUsage(System.err);
//      return -1;
//    }
//    String inputFile = args[0];
//    String outputDir = args[1];
    
    Job job = prepareJob();
    
    return job.waitForCompletion(true) ? 0 : 1;
  }
  
  private Job prepareJob() throws IOException {
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());
    
    String inputFile = "";
    
    if (RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
      IOUtils.deleteRecursively(TRAIN_OUTPUT_PATH);
      inputFile = INPUT_FILE_LOCAL;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = INPUT_FILE_HDFS;
      conf.addResource(new Path(CONFIG_FILE_PATH));
      
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", JAR_PATH);
      
      // Set number of reducers manually (also possible via command line)
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
//      conf.setInt("mapred.tasktracker.reduce.tasks.maximum", 4);
      
      // Delete old output dir
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(TRAIN_OUTPUT_PATH);
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
  
    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(TRAIN_OUTPUT_PATH));
    
    return job;
  }

}
