package de.tuberlin.dima.ml;

import org.apache.mahout.math.Vector;

public interface ClassificationModel {
  
  /**
   * Classify x using the current model.
   * If the model outputs a probability, 
   * the result depends on an threshold (typically 0.5) 
   */
  int classify(Vector x);

}
