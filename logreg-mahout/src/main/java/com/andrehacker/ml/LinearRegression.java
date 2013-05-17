package com.andrehacker.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

/**
 * Inspired by org.apache.mahout.classifier.sgd.TrainLogistic
 * 
 * @author andre
 *
 */
public class LinearRegression implements RegressionModel, ClassificationModel {
  
  public void train(String inputFile, List<String> predictorNames) throws IOException {
    BufferedReader reader = new BufferedReader(MLUtils.open(inputFile));
    
    // Read numeric csv into dense matrix
    CsvReader csv = new CsvReader();
    int rows = 40;
    csv.csvNumericToDenseMatrix(reader, rows, "color", predictorNames, true);
    Matrix data = csv.getData();
    
    // Least squares solution:
    Vector w = pseudoInverse(data).times(csv.getY());
    System.out.println("Learned weigths: " + w);
    
    Matrix confusion = new DenseMatrix(2,2);
    System.out.println("Mean Deviation: " + Validation.computeMeanDeviation(data, csv.getY(), w, this));
    System.out.println("Success-rate: " + Validation.computeSuccessRate(data, csv.getY(), w, this, confusion));
  }
  
  public double predict(Vector x, Vector w, boolean debug) {
    return x.dot(w);
  }

  public int classify(Vector x, Vector w, boolean debug) {
    return (int) Math.round( predict(x, w, debug) );
  }
  
//  public void trainold(String inputFile) throws IOException {
//    
//    Matrix dataTest = new DenseMatrix(3, 2);
//    dataTest.set(0, new double[] {1, 2});
//    dataTest.set(1, new double[] {3, 4});
//    dataTest.set(1, new double[] {5, 6});
//    
//    System.out.println("Size: " + dataTest.numRows() + "x" + dataTest.numCols());
//    
//    // Types: numeric, word or text
//    Map<String, String> typeMap = Maps.newHashMap();
//    typeMap.put("x", "numeric");
//    typeMap.put("y", "numeric");
//    typeMap.put("shape", "numeric");
//    typeMap.put("a", "numeric");
//    typeMap.put("b", "numeric");
//    typeMap.put("c", "numeric");
//
//    CsvRecordFactory csv = new CsvRecordFactory("color", typeMap);
//    csv.defineTargetCategories(Lists.newArrayList("1", "2"));
//    csv.includeBiasTerm(true);
//    BufferedReader reader = new BufferedReader(IoUtils.open(inputFile));
//    csv.firstLine(reader.readLine());
//    //int numFeatures = getNumFeatures(csv);
//    //int numFeatures = typeMap.size() + 2;
//    int numFeatures = 21;   // all features, not just the one we use!
//    System.out.println("Features: " + numFeatures);
//    
//    String line = reader.readLine();
//    int rows = 0;
//    List<Integer> y = Lists.newArrayList();
//    List<Vector> vectors = Lists.newArrayList();
//    while (line != null) {
//      rows++;
//      Vector vec = new DenseVector(numFeatures);
//      y.add( csv.processLine(line, vec) );
//      vectors.add(vec);
//      System.out.println(vec.toString());
//      line = reader.readLine();
//    }
//    System.out.println("Lines: " + rows);
//    
//    // Transform vectors to matrix
//    Matrix data = new DenseMatrix(rows, numFeatures);
//    int row=0;
//    for (Vector vec : vectors) {
//      data.assignRow(row++, vec);
//    }
//    System.out.println(data.viewRow(0));
//    
//    //CSVVectorIterator iter = new CSVVectorIterator(new FileReader(inputFile));
//    //Vector first = iter.next();
//    //first.toString();
//    
//    // SingularValueDecomposition svd = new SingularValueDecomposition(data);
//    //System.out.println(svd.toString());
//  }
  
  /**
   * Computes pseudo inverse using SSVD (stochastical singular value decomposition)
   * See http://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_of_the_SVD
   * 
   * @param data Input Matrix
   * @return pseudo-inverse of data matrix
   */
  private Matrix pseudoInverse(Matrix data) {
    
    SingularValueDecomposition svd = new SingularValueDecomposition(data);
    
    // Use SVD result to compute pseudoinverse of matrix
    // Computation: pinv(data) = V pinv(S) U*
    // where data = U S V* (this is the result of the svd)
    // and   pinv(S) = replacing every non-zero diagonal entry in S by its reciprocal and transposing the resulting matrix
    
    Matrix S = svd.getS();
    
    for (int i=0; i<S.rowSize(); ++i) {
      if (S.get(i, i) != 0) {
        S.set(i, i, 1.0 / S.get(i, i));
      }
    }
    
    return svd.getV().times(S.transpose()).times(svd.getU().transpose());
  }

}
