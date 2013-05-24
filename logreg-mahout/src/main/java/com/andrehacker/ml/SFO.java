package com.andrehacker.ml;

import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

public class SFO {
  
  private CsvReader csvTrain;
  private CsvReader csvTest;
  LogisticRegression logReg;
  IncrementalModel model;
  
  private static final double BIAS_DEFAULT = 1;
  private static final String TARGET_NAME = "color";
  private static final double TARGET_POSITIVE = 2d;
  private static final double TARGET_NEGATIVE = 1d;
  
  // Training Parameters
  private static final int ITERATIONS = 20;
  private static final double INITIAL_WEIGHT = 1d;
  private static final double PENALTY = 1d;
  
  public SFO(String trainingFile, String testFile, List<String> predictorNames) throws Exception {
    this.logReg = new LogisticRegression();
    
    csvTrain = MLUtils.readData(trainingFile, 40, predictorNames, TARGET_NAME);
    csvTrain.normalize();
    csvTrain.normalizeClassLabels(TARGET_POSITIVE, TARGET_NEGATIVE);
    csvTest = MLUtils.readData(testFile, 40, predictorNames, TARGET_NAME);
    csvTest.normalize(csvTrain.getMeans(), csvTrain.getRanges());
    csvTest.normalizeClassLabels(TARGET_POSITIVE, TARGET_NEGATIVE);
    
    // Initiate an empty model, just with bias
    model = new IncrementalModel(BIAS_DEFAULT, csvTrain.getNumPredictors());
  }
  
  /**
   * Keeps a model that can be extended feature by feature
   * Needed for forward feature selection.
   * Keeps track of the used dimensions
   */
  private static class IncrementalModel {
    Vector w;
    private List<Integer> usedDimensions;
    private List<Integer> remainingDimensions;
    
    public IncrementalModel(double biasValue, int numDimensions) {
      // All dimensions are unused at beginning
      usedDimensions = Lists.newArrayList();
      usedDimensions.add(0);    // Bias
      remainingDimensions = Lists.newArrayListWithCapacity(numDimensions);
      for (int i=1; i<=numDimensions; ++i) { remainingDimensions.add(i); }    // considers bias
      
      w = new DenseVector(1);
      w.set(0, biasValue);
    }

    /**
     * Return a copy of the base model with place for one more feature
     */
    public Vector getExtendedModel() {
      Vector extended = new DenseVector(w.size()+1);
      for (int i=0; i<w.size(); ++i) {
        extended.setQuick(i, w.getQuick(i));
      }
      return extended; 
    }
    
    public Vector getW() {
      return w;
    }
    
    public void addDimensionToModel(int d, double weight) {
      remainingDimensions.remove(remainingDimensions.indexOf(d));
      usedDimensions.add(d);
      w = getExtendedModel();
      w.setQuick(w.size()-1, weight);
    }
    
    public List<Integer> getRemainingDimensions() {
      return remainingDimensions;
    }
    
    public List<Integer> getUsedDimensions() {
      return usedDimensions;
    }
  }
  
  public void findBestFeature() {
    
    Validation val = new Validation();
    
    
    // Measure performance of current base model
    Matrix XTestBase = transformMatrix(model.getUsedDimensions(), -1, csvTest.getData());
    val.computeSuccessRate(XTestBase, csvTest.getY(), model.getW(), logReg);
    val.computeMeanDeviation(XTestBase, csvTest.getY(), model.getW(), logReg);
    System.out.println("Base Model performance");
    System.out.println(" - dev:     " + val.getMeanDeviation());
    System.out.println(" - success: " + val.getSuccessRate());
    double baseMeanDeviation = val.getMeanDeviation();

    // TODO Optimization: Always keep an extended Matrix, where the last row can be exchanged (much faster!)
    int bestDimension = 0;
    double bestGain = Double.NEGATIVE_INFINITY;
    double bestWeight = 0;
    double gain = Double.NEGATIVE_INFINITY;
    for (int d : model.getRemainingDimensions()) {
      
      // Get copy of matrix only with relevant dimensions
      Matrix XTrain = transformMatrix(model.getUsedDimensions(), d, csvTrain.getData());
      Matrix XTest = transformMatrix(model.getUsedDimensions(), d, csvTest.getData());
      
      // Train single feature
      System.out.println("Train " + csvTrain.getColumnName(d) + " (d=" + d + ")");
      Vector extendedW = model.getExtendedModel();     // weight vector with place for one more feature
      System.out.println(" - Before: " + extendedW);
      extendedW = logReg.trainNewtonSFO(XTrain, csvTrain.getY(), extendedW, ITERATIONS, INITIAL_WEIGHT, PENALTY);
      System.out.println(" - After:  " + extendedW);

      // Measure performance when adding dimension d to base model
      val.computeSuccessRate(XTest, csvTest.getY(), extendedW, logReg);
      val.computeMeanDeviation(XTest, csvTest.getY(), extendedW, logReg);
      System.out.println(" - dev:     " + val.getMeanDeviation());
      System.out.println(" - success: " + val.getSuccessRate());
      
      // Compute gain
      gain = baseMeanDeviation - val.getMeanDeviation();
      if (gain > bestGain) {
        bestDimension = d;
        bestGain = gain;
        bestWeight = extendedW.getQuick(extendedW.size() - 1);
        System.out.println(" -> New best dimension with gain " + gain);
      }
    }
    System.out.println("=> Best dimension: " + csvTrain.getColumnName(bestDimension) + " (d=" + bestDimension + ")\n");
    
    model.addDimensionToModel(bestDimension, bestWeight);
  }
  
  /**
   * Returns a copy of the input matrix with only relevant columns/dimensions in given order
   * Columns of input:          1 2 3 ... 10
   * Relevant dimensions:       4 2 5
   * Returned matrix columns:   4 2 5
   */
  private Matrix transformMatrix(List<Integer> relevantDimensions, int newDimension, Matrix input) {
    int numColumns = relevantDimensions.size() + (newDimension != -1 ? 1 : 0);
    Matrix copy = new DenseMatrix(csvTrain.getData().rowSize(), numColumns);
    for (int c=0; c<relevantDimensions.size(); ++c) {
      copy.assignColumn(c, input.viewColumn(relevantDimensions.get(c)));
    }
    if (newDimension != -1) {
      // Add the single new dimension
      copy.assignColumn(copy.numCols()-1, input.viewColumn(newDimension));
    }
    return copy;
  }

}
