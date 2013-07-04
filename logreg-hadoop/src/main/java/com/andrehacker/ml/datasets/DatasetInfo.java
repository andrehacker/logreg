package com.andrehacker.ml.datasets;

import java.util.List;
import java.util.Map;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Lists;

/**
 * Provides additional information about a model (usually meta data)
 * E.g. mapping from feature dimension to feature name, number of features or size
 * 
 * For MapReduce we can make static instances of this class,
 *  that are accessible also from Mappers and Reducers
 * 
 */
public class DatasetInfo {

  private long numFeatures=0;
  private long total=0;
  private List<String> predictorNames = Lists.newArrayList();
  private BiMap<Integer, String> labelMap = HashBiMap.create();
  
  /**
   * Builder pattern (effective java, item 2)
   * Makes object immutable and easy to add optional parameters
   */
  public static class Builder {
    
    // Required parameters
    private long numFeatures = 0;
    private long total = 0;
    
    // Optional parameters
    private List<String> predictorNames = null;
    private BiMap<Integer, String> labelMap = null;
    
    public Builder(
      long numFeatures,
      long total) {
      this.numFeatures = numFeatures;
      this.total = total;
    }
    
    public Builder predictorNames(List<String> val) {
      predictorNames = val; return this;
    }
    
    public Builder labelMap(Map<Integer, String> val) {
      labelMap = HashBiMap.create(val); return this;
    }
    
    public DatasetInfo build() {
      return new DatasetInfo(this);
    }
  }
  
  private DatasetInfo(Builder builder) {
    numFeatures = builder.numFeatures;
    total = builder.total;
    predictorNames = builder.predictorNames;
    labelMap = builder.labelMap;
  }
  
  public String getFeatureName(int dimension) {
    if (predictorNames != null && predictorNames.size() > dimension)
      return predictorNames.get(dimension);
    return "unknown feature";
  }
  
  public long getTotal() {
    return total;
  }
  
  public String getLabelNameById(int id) {
    return labelMap.get(id);
  }
  
  public int getLabelIdByName(String name) {
    return labelMap.inverse().get(name);
  }
  
  public long getNumFeatures() {
    return numFeatures;
  }
  
  public void setPredictorNames(List<String> predictorNames) {
    this.predictorNames = predictorNames;
  }
  
}
