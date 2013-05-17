package com.andrehacker.ml;
import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.andrehacker.ml.MLUtils;
import com.mongodb.util.Util;


public class UtilsTest {

  @Test
  public void test() {
    Matrix data = new DenseMatrix(3,2);
    data.set(0, new double[] {1, -3});
    data.set(1, new double[] {2, -1});
    data.set(2, new double[] {3, 1});
    
    Vector mean = MLUtils.meanByColumns(data);
    System.out.println("Mean: " + mean);
    assertEquals(mean.get(0), 2, 0);
    assertEquals(mean.get(1), -1, 0);
    
    Vector range = MLUtils.rangeByColumns(data);
    System.out.println("Range: " + range);
    assertEquals(range.get(0), 2, 0);
    assertEquals(range.get(1), 4, 0);
  }

}
