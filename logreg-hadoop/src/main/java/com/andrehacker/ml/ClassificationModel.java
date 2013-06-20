package com.andrehacker.ml;

import org.apache.mahout.math.Vector;

public interface ClassificationModel {
  
  int classify(Vector x, Vector w);

}
