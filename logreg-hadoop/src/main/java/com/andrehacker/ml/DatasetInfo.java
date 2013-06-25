package com.andrehacker.ml;

import java.util.List;

/**
 * Provides additional information about a model (usually meta data)
 * E.g. mapping from feature dimension to feature name
 * 
 * TODO Add option for unnamed model (e.g. for performance reasons we don't want the term-id -> term mapping)
 */
public class DatasetInfo {
  
  private List<String> predictorNames;
  
  public DatasetInfo(List<String> predictorNames) {
    this.predictorNames = predictorNames;
  }
  
  public String getFeatureName(int dimension) {
    return predictorNames.get(dimension);
  }
  
  public int getNumFeatures() {
    return predictorNames.size();
  }
  
  /**
   * TODO This is wrong, as there might be ids that are not in use
   */
  public int getHighestFeatureId() {
    return predictorNames.size();
  }
}
