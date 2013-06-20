package com.andrehacker.ml;

import java.io.IOException;

import org.apache.mahout.math.Vector;

public interface AbstractVectorReader {

  public int readVector(Vector v, String line);

  public Vector readTarget(int count, String filename, String categoryName) throws IOException;

}
