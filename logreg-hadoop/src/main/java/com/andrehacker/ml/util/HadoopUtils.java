package com.andrehacker.ml.util;

import org.apache.hadoop.conf.Configuration;

public class HadoopUtils {

  public static boolean detectLocalMode(Configuration conf) {
    return conf.get("mapred.job.tracker").equals("local");
  }
  
}
