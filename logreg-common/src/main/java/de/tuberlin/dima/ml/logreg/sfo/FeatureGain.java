package de.tuberlin.dima.ml.logreg.sfo;

import com.google.common.primitives.Doubles;

public class FeatureGain implements Comparable<FeatureGain> {
  private int dimension;
  private double gain; // currently log-likelihood gain

  public FeatureGain(int dimension, double gain) {
    this.dimension = dimension;
    this.gain = gain;
  }

  @Override
  public int compareTo(FeatureGain other) {
    return Doubles.compare(this.gain, other.gain);
  }

  public int getDimension() {
    return dimension;
  }

  public double getGain() {
    return gain;
  }
}
