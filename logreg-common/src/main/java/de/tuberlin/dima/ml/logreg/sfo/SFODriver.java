package de.tuberlin.dima.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface SFODriver {

  public List<FeatureGain> computeGainsSFO(int dop) throws Exception;

  public void addBestFeature() throws IOException;

  public void addNBestFeatures(int n) throws IOException;
  
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
  
}
