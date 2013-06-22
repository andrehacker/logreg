package com.andrehacker.ml;

import java.util.List;

/**
 * Provides additional information about a model (usually meta data)
 * E.g. mapping from feature dimension to feature name  
 */
public class ModelInfo {
  
  private List<String> predictorNames;
  
  public ModelInfo(List<String> predictorNames) {
    this.predictorNames = predictorNames;
  }
  
  public String getFeatureName(int dimension) {
    return predictorNames.get(dimension);
  }
  
  public int getNumFeatures() {
    return predictorNames.size();
  }
}
