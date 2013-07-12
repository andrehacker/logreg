package de.tuberlin.dima.ml.logreg.iterative;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;

import edu.stanford.nlp.optimization.QNMinimizer;

/**
 * L-BFGS with MR implementation of DiffFunction.
 */
public class LBFGSDriver extends Configured implements Tool {

  private GradientJob gradientJob;
  private TrainingErrorJob trainingErrorJob;
  private double eps;
  private final int maxIterations;
  private double[] initial;

  public LBFGSDriver(TrainingErrorJob trainingErrorJob, GradientJob gradientJob, 
      double eps, double[] initial) {

    this(trainingErrorJob, gradientJob, eps, initial, 0);
  }
  
  public LBFGSDriver(TrainingErrorJob trainingErrorJob, GradientJob gradientJob, 
      double eps, double[] initial, int maxIterations) {

    this.trainingErrorJob = trainingErrorJob;
    this.gradientJob = gradientJob;
    this.eps = eps;
    this.initial = initial;
    this.maxIterations = maxIterations;
  }

  @Override
  public int run(String[] args) throws Exception {
    
    LBFGSDiffFunction f = new LBFGSDiffFunction(this.trainingErrorJob, this.gradientJob);
    
    QNMinimizer qn = new QNMinimizer(15, true);
    
    double[] model;
    
    if (this.maxIterations == 0) {
      model = qn.minimize(f, this.eps, this.initial);      
    } else {
      model = qn.minimize(f, this.eps, this.initial, this.maxIterations);            
    }
    
    return 0;
  }
}