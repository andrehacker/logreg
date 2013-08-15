package de.tuberlin.dima.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;

public interface SFODriver {

  public List<FeatureGain> computeGainsSFO(int dop) throws Exception;

  public void addBestFeature() throws IOException;

  public void addNBestFeatures(int n) throws IOException;
  
  public void retrainBaseModel();

  public List<FeatureGain> getGains();
  
}
