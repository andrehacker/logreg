package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.mapred.GlobalSettings;
import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import de.tuberlin.dima.ml.util.MathUtil;

public class SFOReducer extends
    Reducer<IntWritable, SFOIntermediateWritable, IntWritable, DoubleWritable> {

  private static final AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(SFOReducer.class.getName()), GlobalSettings.LOG_LEVEL);

  private static final int MAX_ITERATIONS = 5;

  private static final double LAMBDA = 0;
  private static final double TOLERANCE = 10E-7;

  private static final int DEBUG_DIMENSION = -1;

  /**
   * Notes: To iterate multiple times over data, we cache all data on heap For
   * each dimension, this stores as many objects in memory as documents
   * containing this word Assumption: Should always fit into memory
   * 
   * We could avoid this by learning with Online SGD in one pass, but for many
   * dimensions we will have few data and I am unsure if this works well
   */
  @Override
  public void reduce(IntWritable dim, Iterable<SFOIntermediateWritable> values,
      Context context) throws IOException, InterruptedException {

    // log.debug("Reducer for d=" + dim.get());

    List<SFOIntermediateWritable> cache = Lists.newArrayList();

    // TODO Improvement: Run loop until convergence
    double betad = 0;
    int iteration = 0;
    double lastUpdate = 0;
    boolean converged = false;
    while ((++iteration <= MAX_ITERATIONS) && !converged) {

      double batchGradient = 0;
      double batchGradientSecond = 0;
      double debugSumPi = 0;
      Iterable<SFOIntermediateWritable> currentIterable = (iteration == 1 ? values
          : cache);
      for (SFOIntermediateWritable element : currentIterable) {

        if (iteration == 1) {
          cache.add(new SFOIntermediateWritable(element)); // copy via copy
                                                           // constructor
        }

        /*
         * Here we reverse engineer the value of x*w, instead of transmitting it
         * directly. The used formula is the same as in the binning optimization
         * section, maybe it related to this
         * 
         * TODO Improvement: Why not transfer beta_d * x_id?
         */
        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));

        /*
         * Compute the 1st and 2nd derivate
         * 
         * Handle Underflows: If we receive underflow then we would have been
         * close to zero anyhow, so nothing to do here
         * 
         * Handle Overflows: If we receive overflow, we have a problem, because
         * we can not calculate with Infinity value. We should avoid this
         * (regularization?)
         * 
         * TODO Bug: Why does Singh not use (xDotw + element.getXid() * betad)??
         * This is the general derivation!
         */
        double exponent = xDotw + (element.getXid() * betad);
        double piNew = LogRegMath.logisticFunction(exponent);
        if (!MathUtil.checkDouble(piNew, true)) {
          log.debug("- INVALID RESULT: d=" + dim.get() + " iteration="
              + iteration + " cacheSize=" + cache.size() + " xDotw=" + xDotw
              + " x_id=" + element.getXid() + " betad=" + betad
              + " last-update=" + lastUpdate);
        }
        debugSumPi += piNew;

        batchGradient += LogRegSFOTraining.derivateL2SFO(element.getXid(),
            piNew, element.getLabel(), LAMBDA, betad);
        batchGradientSecond += LogRegSFOTraining.derivateSecondL2SFO(
            element.getXid(), piNew, LAMBDA);
      }

      /*
       * Newton update
       * 
       * If first and/or second derivate is zero we probably already converged
       * to the optimum and should stop (otherwise we divide by zero)
       * 
       * TODO Why does Ng use another error-function (and derivate) where we
       * divide by N?
       */
      if (batchGradientSecond == 0) {
        lastUpdate = 0; // Avoid division by zero (NaN), no update needed
      } else {
        lastUpdate = (batchGradient / batchGradientSecond);
        betad -= lastUpdate;
      }

      /*
       * Check for conversion
       */
      if (Math.abs(lastUpdate) < TOLERANCE) {
        converged = true;
      }

      if (dim.get() == DEBUG_DIMENSION)
        log.debug("- it " + iteration + ": grad: " + batchGradient
            + " gradSecond: " + batchGradientSecond + " new beta_d: " + betad
            + " sumPi: " + debugSumPi);
    }

    // Write trained coefficient
    context.write(dim, new DoubleWritable(betad));

    // log.debug("- Processed " + cache.size() + " records, new beta: " +
    // betad);
  }

}
