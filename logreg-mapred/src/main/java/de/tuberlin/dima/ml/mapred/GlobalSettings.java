package de.tuberlin.dima.ml.mapred;

import org.apache.log4j.Level;

/**
 * Static settings, for various Hadoop jobs
 * 
 * The goal is to remove all those global static settings and load them at runtime
 */
public class GlobalSettings {
  
  public static final String CONFIG_FILE_PATH = "core-site-local.xml";
  // static final String CONFIG_FILE_PATH = "core-site-pseudo-distributed.xml";

  public static final Level LOG_LEVEL = Level.DEBUG;
  
  //TODO What to set this to? Why does Singh not train it?
  public static final double INTERCEPT = 1;
  
  // --------- Settings for execution in a cluster ------------
  public static final String JAR_PATH = "target/logreg-0.0.1-SNAPSHOT-job.jar";
  
}
