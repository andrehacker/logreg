package com.andrehacker.ml.sfo;

import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

/**
 * Keeps a model that can be extended feature by feature
 * Needed for forward feature selection.
 * Keeps track of the used features (base model) and the unused/remaining features
 * 
 * Assumptions:
 * - We ignore the interception term.
 * 
 * TODO Maybe I don't need unusedDimensions at all, because they are just the rest?!
 * TODO Tradeoff: Either let w also include unused features, or extract only the used
 * V1 (big w): No costs for extraction of used features in w. Requires knowledge of total features. Might take longer to multiply (depends on implementation of sparse vector)
 * V2 (small w): Use DenseVector. Fast to multiply (if basemodel is small). If basemodel is big
 */
class IncrementalModel {
  
  private Vector w;
  private List<Integer> usedDimensions;
  private List<Integer> unusedDimensions;

  /**
   * Create empty model where all dimensions are unused
   */
  public IncrementalModel(int numDimensions) {
    usedDimensions = Lists.newArrayList();
    
    unusedDimensions = Lists.newArrayListWithCapacity(numDimensions);
    for (int i=0; i<=numDimensions; ++i) { unusedDimensions.add(i); }
    
    w = new RandomAccessSparseVector(numDimensions);
  }
  
  /**
   * Create empty model where all dimensions are unused
   */
  public IncrementalModel(List<Integer> remainingDimensions, int numDimensions) {
    usedDimensions = Lists.newArrayList();
    this.unusedDimensions = remainingDimensions;

    w = new RandomAccessSparseVector(numDimensions);
  }
  
  public IncrementalModel(List<Integer> usedDimensions, List<Integer> remainingDimensions, Vector w) {
    this.usedDimensions = usedDimensions;
    this.unusedDimensions = remainingDimensions;
    this.w = w;
  }
  
  public void addDimensionToModel(int d, double weight) {
    unusedDimensions.remove(unusedDimensions.indexOf(d));
    usedDimensions.add(d);
    w.setQuick(d, weight);
  }
  
  public Vector getW() {
    return w;
  }
  
  public List<Integer> getUnusedDimensions() {
    return unusedDimensions;
  }
  
  public List<Integer> getUsedDimensions() {
    return usedDimensions;
  }
}