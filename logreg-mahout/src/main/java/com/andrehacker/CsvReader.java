package com.andrehacker;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;

public class CsvReader {

  private static final Splitter splitter = Splitter.on(',').trimResults(CharMatcher.is('"'));
  
  private static final int BIAS_DEFAULT = 1;
  
  private Map<String, String> predictorsAndTypes;
  private List<Integer> predictorIndices;
  private int targetIndex;
  private List<String> variableNames;
  private Matrix data;
  private Vector y;
  
  /**
   * Read a numeric csv to dense Matrix representation
   * For small matrices (has to fit into memory)
   * 
   * Result can be obtained using getters 
   * 
   * @param reader
   * @return
   * @throws IOException 
   */
  public boolean csvNumericToDenseMatrix(BufferedReader reader, int rows, String targetName, List<String> predictorNames, boolean addBias) throws IOException {
    
    // First line contains names
    variableNames = Lists.newArrayList(splitter.split(reader.readLine()));
    // Find predictors and store indices
    Integer index = -1;
    predictorIndices = Lists.newArrayList();
    for (String predictor : predictorNames) {
      index = variableNames.indexOf(predictor);
      if (index != -1) {
        predictorIndices.add(index);
      } else {
        System.out.println("Could not find predictor in header: " + predictor);
      }
    }
    Collections.sort(predictorIndices);
    
    // Find target name
    targetIndex = variableNames.indexOf(targetName);
    if (targetIndex == -1) {
      System.out.println("Could not find target in header: " + targetName);
      return false;
    }
    
    // Construct empty matrix (with or without bias in first column)
    int numColumns = predictorIndices.size() + (addBias ? 1 : 0);
    data = new DenseMatrix(rows, numColumns);
    
    // Read lines and fill matrix
    // Read only the columns defined in predictorIndices (which is sorted)
    String line;
    y = new DenseVector(rows);
    int id;
    int nextPredictorId;
    int vecPosition;
    int row = 0;
    while ((line = reader.readLine()) != null) {
      id = 0;
      nextPredictorId = 0;
      vecPosition = 0;
      double[] vec = new double[numColumns];
      if (addBias) {
        vec[0] = BIAS_DEFAULT;
        vecPosition = 1;
      }
      for (String field : splitter.split(line)) {
        if (targetIndex == id) {
          // This is the target value, add to y
          y.set(row, Double.parseDouble(field));
        }
        if (predictorIndices.get(nextPredictorId) == id) {
          // read this
          vec[vecPosition] = Double.parseDouble(field);
          ++nextPredictorId;
          ++vecPosition;
          if (nextPredictorId >= getNumPredictors()) { 
            break;
          }
        }
        data.set(row, vec);
        ++id;
      }
      ++row;
    }
    
    return true;
    
  }

  public Vector getY() {
    return y;
  }
  
  public Matrix getData() {
    return data;
  }
  
  public int getNumPredictors() {
    return predictorIndices.size();
  }
}
