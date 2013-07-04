package com.andrehacker.ml.logreg.sfo;

import org.apache.log4j.Level;

import com.andrehacker.ml.datasets.DatasetInfo;
import com.andrehacker.ml.datasets.DonutDatasetInfo;

/**
 * Static settings for the SFO Hadoop jobs.
 * 
 * We can refactor this at a later time to be loaded dynamically on runtime.
 * This would require to pass the arguments to all the tasks (map/reduce)
 * via job configuration, distributed cache or hdfs. 
 */
public class GlobalJobSettings {
  
  // TODO Minor: Remove this redundancy
  static final boolean RUN_LOCAL_MODE = true;
  static final String CONFIG_FILE_PATH = "core-site-local.xml";
//  static final String CONFIG_FILE_PATH = "core-site-pseudo-distributed.xml";

  static final Level LOG_LEVEL = Level.WARN;

  static final String BASE_MODEL_PATH = "sfo-base-model.seq";
  
  static DatasetInfo datasetInfo = DonutDatasetInfo.get();

  //TODO What to set this to? Why does Singh not train it?
  static final double INTERCEPT = 1;
  
  
  // --------- Settings for execution in a cluster ------------
  
  static final String JAR_PATH = "target/logreg-0.0.1-SNAPSHOT-job.jar";
  
}
