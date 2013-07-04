package com.andrehacker.ml.util;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Wrapper class for Logger
 * Prints to stdout if we run in local mode (makes debugging easier)
 * 
 * Would be nice to get rid of this class and use sl4j only  
 */
public class AdaptiveLogger {
  
  private boolean runLocal; // True in standalone/local mode

  private Logger logger;
  private Level level;
  
  public AdaptiveLogger(boolean runLocal, Logger logger, Level level) {
    this.runLocal = runLocal;
    this.logger = logger;
    setLevel(level);
  }
  
  public AdaptiveLogger(boolean runLocal, Logger logger) {
    this.runLocal = runLocal;
    this.logger = logger;
  }

  public void debug(String line) {
    if (runLocal) {
      if (level == Level.TRACE || level == Level.DEBUG || level == Level.ALL)
        System.out.println(line);
    } else {
      logger.debug(line);
    }
  }

  public void setLevel(Level level) {
    this.level = level;
    logger.setLevel(level);
  }
  
}
