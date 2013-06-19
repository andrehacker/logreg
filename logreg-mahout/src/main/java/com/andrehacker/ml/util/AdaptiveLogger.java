package com.andrehacker.ml.util;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Class for Logger
 * Prints to stdout if we run in local mode (makes debugging easier)  
 */
public class AdaptiveLogger {
  
  private boolean runLocal; // True in standalone/local mode

  private Logger logger;
  
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
      System.out.println(line);
    } else {
      logger.debug(line);
    }
  }

  public void setLevel(Level level) {
    logger.setLevel(level);
  }
  
}
