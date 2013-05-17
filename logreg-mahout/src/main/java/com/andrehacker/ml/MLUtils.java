package com.andrehacker.ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;

public class MLUtils {
  
  /**
   * Try to open file for reading from Resources.
   * If there is no such resource, try to read from filesystem. 
   * 
   * @param inputFile
   * @return
   * @throws IOException
   */
  static BufferedReader open(String inputFile) throws IOException {
    InputStream in;
    try {
      in = Resources.getResource(inputFile).openStream();
    } catch (IllegalArgumentException e) {
      in = new FileInputStream(new File(inputFile));
    }
    return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
  }
  
  static Vector meanByColumns(Matrix m) {
    Vector sums = m.aggregateColumns(new VectorFunction() {
      public double apply(Vector col) {
        return col.aggregate(Functions.PLUS, Functions.IDENTITY);
      }
    });
    return sums.divide(m.numRows());
  }
  
  static Vector rangeByColumns(Matrix m) {
    Vector min = m.aggregateColumns(new VectorFunction() {
      public double apply(Vector f) {
        return f.minValue();
      }
    });
    Vector max = m.aggregateColumns(new VectorFunction() {
      public double apply(Vector f) {
        return f.maxValue();
      }
    });
    return max.minus(min);
  }
  
  private static void printDimensions(Matrix matrix) {
    System.out.println("Size: " + matrix.rowSize() + "x" + matrix.columnSize());
  }

}
