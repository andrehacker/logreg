package de.tuberlin.dima.ml.util;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Wrapper class for Logger
 * 
 * Detects if we run in localmode and logs to System.out in this case
 * Eclipse will highlight the output then (easy to see)
 * Otherwise it logs to log4j
 * 
 * TODO Finish! Need a reference to Configuration instance to detect local mode.
 * 
 * Would be nice to get rid of this class and use sl4j only  
 */
public class AdaptiveLogger {
  
  private boolean runLocal = true;

  private Logger logger;
  private Level level;
  
  public AdaptiveLogger(Logger logger, Level level) {
    this.logger = logger;
    setLevel(level);
  }
  
  public AdaptiveLogger(Logger logger) {
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
