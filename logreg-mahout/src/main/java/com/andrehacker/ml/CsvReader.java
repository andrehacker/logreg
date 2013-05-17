package com.andrehacker.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
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
  
  // predictorIndices and predictorNames have same order!
  private boolean addBias;
  private List<Integer> predictorIndices;
  Map<Integer, String> columnToNameMap;
  private int targetIndex;
  private List<String> allVariables;
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
    
    this.addBias = addBias;

    // First line contains names
    allVariables = Lists.newArrayList(splitter.split(reader.readLine()));
    // Find predictors and store indices
    this.columnToNameMap = new HashMap<Integer, String>();
    Integer index = -1;
    predictorIndices = Lists.newArrayList();
    int curTargetColumn = 0;    // column where to store in the matrix
    for (int id = 0; id < allVariables.size() ; ++id) {
      if ((index = predictorNames.indexOf(allVariables.get(id))) != -1) {
        columnToNameMap.put(curTargetColumn + (addBias?1:0), predictorNames.get(index));
        predictorIndices.add(id);
        curTargetColumn++;
      }
    }
    
//  // First line contains names
//  allVariables = Lists.newArrayList(splitter.split(reader.readLine()));
//  // Find predictors and store indices
//  this.columnToNameMap = new HashMap<Integer, String>();
//  Integer index = -1;
//  predictorIndices = Lists.newArrayList();
//  for (String predictor : predictorNames) {
//    index = allVariables.indexOf(predictor);
//    if (index != -1) {
//      predictorIndices.add(index);
//      columnToNameMap.put(index, predictor);
//    } else {
//      System.out.println("Could not find predictor in header: " + predictor);
//      columnToNameMap.put(-1, predictor);
//    }
//  }
//  Collections.sort(predictorIndices);
    
    // Find target name
    targetIndex = allVariables.indexOf(targetName);
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
    int currentColumn;
    int row = 0;
    while ((line = reader.readLine()) != null) {
      id = 0;
      nextPredictorId = 0;
      currentColumn = 0;
      if (addBias) {
        data.set(row, 0, BIAS_DEFAULT);
        currentColumn = 1;
      }
      for (String field : splitter.split(line)) {
        if (targetIndex == id) {
          // This is the target value, add to y
          y.set(row, Double.parseDouble(field));
        }
        if (predictorIndices.get(nextPredictorId) == id) {
          // this is a predictor value
          data.set(row, currentColumn, Double.parseDouble(field));
          ++nextPredictorId;
          ++currentColumn;
          if (nextPredictorId >= getNumPredictors()) {
            break;
          }
        }
        ++id;
      }
      ++row;
    }
    
    return true;
    
  }

  public String getColumnName(int i) {
    if (i == 0 && addBias) return "Bias";
    return columnToNameMap.get(i);
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
