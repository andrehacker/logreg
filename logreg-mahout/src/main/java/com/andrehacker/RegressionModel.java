package com.andrehacker;

import org.apache.mahout.math.Vector;

public interface RegressionModel {
  
  double hypothesis(Vector x, Vector w, boolean debug);

}
