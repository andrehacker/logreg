package com.andrehacker.ml;

import java.io.BufferedReader;
import java.util.List;
import java.util.Map;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class CsvReader {

  // TODO: Allow spaces as separator: "a", "b", ...
  private static final Splitter splitter = Splitter.on(',').trimResults(CharMatcher.is('"'));
  
  private static final int BIAS_DEFAULT = 1;
  private static final String BIAS_NAME = "BIAS";
  
  // predictorIndices and predictorNames have same order!
  private boolean addBias;
//  private List<Integer> predictorIndices;
  private Map<Integer, Integer> internalToFileColumnMap;    // maps columns of internal matrix to file columns
//  private Map<Integer, String> columnInternalToNameMap;
  private List<String> predictorNames;
  private List<String> allVariables;
  private Matrix data;
  private Vector y;
  
  // Normalization variables
  Vector means;
  Vector ranges;
  
  /**
   * Read a numeric csv to dense Matrix representation
   * For small matrices (has to fit into memory)
   * 
   * Result can be obtained using getters
   * @throws Exception 
   */
  public boolean numericToDenseMatrix(BufferedReader reader, int rows, String targetName, List<String> predictorNames, boolean addBias) throws Exception {
    
    this.addBias = addBias;
    this.predictorNames = predictorNames;
    this.internalToFileColumnMap = Maps.newHashMap();

    // Find predictors (in first line) and store indices
    allVariables = Lists.newArrayList(splitter.split(reader.readLine()));
    int fileCol = -1;
    for (int predictorNum=0; predictorNum<predictorNames.size(); ++predictorNum) {
      if ((fileCol = allVariables.indexOf(predictorNames.get(predictorNum))) != -1) {
        internalToFileColumnMap.put(predictorNum+(addBias?1:0), fileCol);
      } else {
        throw new Exception("Predictor " + predictorNames.get(predictorNum) + " not found in csv header");
      }
    }
    
    // Find target name
    int targetIndex = allVariables.indexOf(targetName);
    if (targetIndex == -1) {
      throw new Exception("Could not find target in header: " + targetName);
    }
    
    // Construct empty matrix (with or without bias in first column)
    int numColumns = predictorNames.size() + (addBias ? 1 : 0);
    data = new DenseMatrix(rows, numColumns);
    
    // Read lines and fill matrix
    // Read only the columns defined in predictorIndices (which is sorted)
    String line;
    y = new DenseVector(rows);
    int row = 0;
    while ((line = reader.readLine()) != null) {
      if (addBias) {
        data.set(row, 0, BIAS_DEFAULT);
      }
      List<String> fields = Lists.newArrayList(splitter.split(line));
      // get target value
      y.set(row, Double.parseDouble(fields.get(targetIndex)));

      for (int internalCol=0+(addBias?1:0); internalCol<predictorNames.size()+(addBias?1:0); ++internalCol) {
        data.set(row, 
            internalCol, 
            Double.parseDouble(fields.get(internalToFileColumnMap.get(internalCol))));
      }
      ++row;
    }
    
    return true;
  }
  
//  /**
//   * Read a numeric csv to dense Matrix representation
//   * For small matrices (has to fit into memory)
//   * 
//   * Result can be obtained using getters
//   */
//  public boolean numericToDenseMatrixOld(BufferedReader reader, int rows, String targetName, List<String> predictorNames, boolean addBias) throws IOException {
//    
//    this.addBias = addBias;
//
//    // First line contains names
//    allVariables = Lists.newArrayList(splitter.split(reader.readLine()));
//    // Find predictors and store indices
//    this.columnInternalToNameMap = new HashMap<Integer, String>();
//    Integer index = -1;
//    List<Integer> predictorIndices = Lists.newArrayList();
//    int curTargetColumn = 0;    // column where to store in the matrix
//    for (int id = 0; id < allVariables.size() ; ++id) {
//      if ((index = predictorNames.indexOf(allVariables.get(id))) != -1) {
//        columnInternalToNameMap.put(curTargetColumn + (addBias?1:0), predictorNames.get(index));
//        predictorIndices.add(id);
//        curTargetColumn++;
//      }
//    }
//    
//    // Find target name
//    targetIndex = allVariables.indexOf(targetName);
//    if (targetIndex == -1) {
//      System.out.println("Could not find target in header: " + targetName);
//      return false;
//    }
//    
//    // Construct empty matrix (with or without bias in first column)
//    int numColumns = predictorIndices.size() + (addBias ? 1 : 0);
//    data = new DenseMatrix(rows, numColumns);
//    
//    // Read lines and fill matrix
//    // Read only the columns defined in predictorIndices (which is sorted)
//    String line;
//    y = new DenseVector(rows);
//    int id;
//    int nextPredictorId;
//    int currentColumn;
//    int row = 0;
//    while ((line = reader.readLine()) != null) {
//      id = 0;
//      nextPredictorId = 0;
//      currentColumn = 0;
//      if (addBias) {
//        data.set(row, 0, BIAS_DEFAULT);
//        currentColumn = 1;
//      }
//      for (String field : splitter.split(line)) {
//        if (targetIndex == id) {
//          // This is the target value, add to y
//          y.set(row, Double.parseDouble(field));
//        }
//        if (predictorIndices.get(nextPredictorId) == id) {
//          // this is a predictor value
//          data.set(row, currentColumn, Double.parseDouble(field));
//          ++nextPredictorId;
//          ++currentColumn;
//          if (nextPredictorId >= getNumPredictors()) {
//            break;
//          }
//        }
//        ++id;
//      }
//      ++row;
//    }
//    
//    return true;
//  }
  
  public void normalize(Vector means, Vector ranges) {
    this.means = means;
    this.ranges = ranges;

    // TODO: Handle the case where range is 0
    for (int col=1; col<data.numCols(); ++col) {
      Vector newCol = data.viewColumn(col).assign(Functions.MINUS, means.get(col)).divide(ranges.get(col));
      data.assignColumn(col, newCol);
    }
  }
  
  public void normalize() {
    normalize(MLUtils.meanByColumns(data), MLUtils.rangeByColumns(data));
  }
  
  public Vector getMeans() {
    return means;
  }
  
  public Vector getRanges() {
    return ranges;
  }

  /**
   * Map target values to 1 (positive) and 0 (negative)
   * @param targetPositive
   * @param targetNegative
   */
  public void normalizeClassLabels(double targetPositive, double targetNegative) {
    Vector yNew = new DenseVector(y.size());
    for(int i=0; i<y.size(); ++i) {
      yNew.setQuick(i, (y.getQuick(i) == targetPositive) ? 1 : 0 );
    }
    y = yNew;
  }

  public String getColumnName(int i) {
    if (i == 0 && addBias) return BIAS_NAME;
//    return columnInternalToNameMap.get(i);
    return predictorNames.get(i-(addBias?1:0));
  }

  public Vector getY() {
    return y;
  }
  
  public Matrix getData() {
    return data;
  }
  
  public int getNumPredictors() {
//    return predictorIndices.size();
    return data.numCols()-(addBias?1:0);
  }
}
