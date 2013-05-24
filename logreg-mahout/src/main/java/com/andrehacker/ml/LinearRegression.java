package com.andrehacker.ml;

import java.io.BufferedReader;
import java.util.List;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Inspired by org.apache.mahout.classifier.sgd.TrainLogistic
 * 
 * @author andre
 *
 */
public class LinearRegression implements RegressionModel, ClassificationModel {
  
  public void train(String inputFile, List<String> predictorNames) throws Exception {
    BufferedReader reader = new BufferedReader(MLUtils.open(inputFile));
    
    // Read numeric csv into dense matrix
    CsvReader csv = new CsvReader();
    int rows = 40;
    csv.numericToDenseMatrix(reader, rows, "color", predictorNames, true);
    Matrix data = csv.getData();
    
    // Least squares solution:
    Vector w = MLUtils.pseudoInversebySVD(data).times(csv.getY());
    System.out.println("Learned weigths: " + w);

    // TODO: Update this
//    Matrix confusion = new DenseMatrix(2,2);
//    System.out.println("Mean Deviation: " + Validation.computeMeanDeviation(data, csv.getY(), w, this));
//    System.out.println("Success-rate: " + Validation.computeSuccessRate(data, csv.getY(), w, this, confusion));
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
  
}
