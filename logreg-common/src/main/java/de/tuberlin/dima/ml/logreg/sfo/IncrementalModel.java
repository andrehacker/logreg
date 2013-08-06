package de.tuberlin.dima.ml.logreg.sfo;

import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.ClassificationModel;
import de.tuberlin.dima.ml.RegressionModel;
import de.tuberlin.dima.ml.logreg.LogRegMath;

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
 * V2 (small w): Use DenseVector. Fast to multiply (if basemodel is small).
 */
public class IncrementalModel implements RegressionModel, ClassificationModel {
  
  private Vector w;
  private List<Integer> usedDimensions;

  /**
   * Create empty model where all dimensions are unused
   */
  public IncrementalModel(int numDimensions) {
    usedDimensions = Lists.newArrayList();
    w = new RandomAccessSparseVector(numDimensions);
  }
  
  public IncrementalModel(List<Integer> usedDimensions, Vector w) {
    this.usedDimensions = usedDimensions;
    this.w = w;
  }
  
  public void addDimensionToModel(int d, double weight) {
    usedDimensions.add(d);
    w.setQuick(d, weight);
  }
  
  public Vector getW() {
    return w;
  }
  
  public boolean isFeatureUsed(int id) {
    return usedDimensions.contains(id);
  }
  
  public List<Integer> getUsedDimensions() {
    return usedDimensions;
  }

  @Override
  public int classify(Vector x) {
    return LogRegMath.classify(x, w, 0, 0.5);
  }

  @Override
  public double predict(Vector x) {
    return LogRegMath.predict(x, w);
  }

  @Override
  public double predict(Vector x, double intercept) {
    return LogRegMath.predict(x, w, intercept);
  }
  
}