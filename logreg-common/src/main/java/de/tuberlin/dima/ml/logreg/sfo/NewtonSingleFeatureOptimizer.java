package de.tuberlin.dima.ml.logreg.sfo;

import java.util.List;

import de.tuberlin.dima.ml.logreg.LogRegMath;

public class NewtonSingleFeatureOptimizer {
  
  public static double train(List<Record> cache, int maxIterations, double lambda, double tolerance) {
    double betad = 0;
    int iteration = 0;
    double lastUpdate = 0;
    boolean converged = false;
    while ((++iteration <= maxIterations) && !converged) {

      double batchGradient = 0;
      double batchGradientSecond = 0;
      for (Record element : cache ) {

        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));

        double piNew = LogRegMath.logisticFunction(
            xDotw + (element.getXid() * betad));

        batchGradient += LogRegSFOTraining.derivateL2SFO(element.getXid(),
            piNew, element.getYi(), lambda, betad);
        batchGradientSecond += LogRegSFOTraining.derivateSecondL2SFO(
            element.getXid(), piNew, lambda);
      }

      if (batchGradientSecond == 0) {
        lastUpdate = 0;
      } else {
        lastUpdate = (batchGradient / batchGradientSecond);
        betad -= lastUpdate;
      }

      if (Math.abs(lastUpdate) < tolerance) {
        converged = true;
      }
    }
    return betad;
  }
  
  public static final class Record {
    
    private double xid;
    private int yi;
    private double pi;
    
    public Record(double xid, int yi, double pi) {
      this.xid = xid;
      this.yi = yi;
      this.pi = pi;
    }
    
    public double getXid() { return xid; }
    public int getYi() { return yi; }
    public double getPi() { return pi; }
  }

}
