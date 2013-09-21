package de.tuberlin.dima.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface SFODriver {

  public List<FeatureGain> computeGains(int fullDop) throws Exception;

  /**
   * This is equal to computeGains(dop) if iterations = 1
   */
  public List<FeatureGain> forwardFeatureSelection(int fullDop, int iterations, int addPerIteration) throws Exception;

  public void addBestFeatures(int numFeatures) throws IOException;
  
  public void retrainBaseModel();

  public List<FeatureGain> getGains();
  
  /**
   * @return elapsed time (wall-clock time) for the run of computeGainSFO
   */
  public long getLastWallClockTime();

  /**
   * @return All available counters (also timers), e.g. the time for training, testing, ...
   */
  public Map<String, Long> getAllCounters();
  
  /**
   * Reset the model to the initial (empty) model
   */
  public void resetModel();
  
}
