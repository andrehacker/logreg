package com.andrehacker;

import org.apache.mahout.math.Vector;

public interface ClassificationModel {
  
  int classify(Vector x, Vector w, boolean debug);

}
