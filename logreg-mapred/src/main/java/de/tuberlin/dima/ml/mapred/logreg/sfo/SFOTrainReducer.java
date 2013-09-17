package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.logreg.sfo.LogRegSFOTraining;
import de.tuberlin.dima.ml.mapred.GlobalSettings;
import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;

public class SFOTrainReducer extends
    Reducer<IntWritable, SFOIntermediateWritable, IntWritable, DoubleWritable> {

  private static final AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(SFOTrainReducer.class.getName()), GlobalSettings.LOG_LEVEL);
  
//  private double newtonTolerance;
//  private int newtonMaxIterations;
//  private double regularization;

  private int newtonMaxIterations;
  private double newtonTolerance;
  private double regularization;

  private static final int DEBUG_DIMENSION = -1;
  
  @Override
  protected void setup(Context context)
      throws IOException, InterruptedException {
    super.setup(context);
    this.newtonMaxIterations = context.getConfiguration().getInt(SFOTrainJob.CONF_KEY_NEWTON_MAX_ITERATIONS, -1);
    if (this.newtonMaxIterations == -1) {
      throw new RuntimeException("Value for the configuration parameter CONF_KEY_NEWTON_MAX_ITERATIONS is not defined, please set it in plan assembler");
    }
    // TODO: BUG: parameters.getDouble always returns default value
    this.newtonTolerance = context.getConfiguration().getDouble(SFOTrainJob.CONF_KEY_NEWTON_TOLERANCE, -1);
    if (this.newtonTolerance == -1) {
      throw new RuntimeException("Value for the configuration parameter CONF_KEY_NEWTON_TOLERANCE is not defined, please set it in plan assembler");
    }
    this.regularization = context.getConfiguration().getDouble(SFOTrainJob.CONF_KEY_REGULARIZATION, -1);
    if (this.regularization == -1) {
      throw new RuntimeException("Value for the configuration parameter CONF_KEY_REGULARIZATION is not defined, please set it in plan assembler");
    }
  }

  /**
   * Notes: To iterate multiple times over data, we cache all data on heap. For
   * each dimension, this stores as many objects in memory as documents
   * containing this word. Assumption: Should always fit into memory
   * 
   * We could avoid this by learning with Online SGD in one pass, but for many
   * dimensions we will have few data and I am not sure if this works well
   */
  @Override
  public void reduce(IntWritable dim, Iterable<SFOIntermediateWritable> values,
      Context context) throws IOException, InterruptedException {

    // log.debug("Reducer for d=" + dim.get());

    List<SFOIntermediateWritable> cache = Lists.newArrayList();

    double betad = 0;
    int iteration = 0;
    double lastUpdate = 0;
    boolean converged = false;
    while ((++iteration <= newtonMaxIterations) && !converged) {

      double batchGradient = 0;
      double batchGradientSecond = 0;
      Iterable<SFOIntermediateWritable> currentIterable = (iteration == 1 ? values : cache);
      for (SFOIntermediateWritable element : currentIterable) {

        if (iteration == 1) {
          cache.add(new SFOIntermediateWritable(element));
        }

        /*
         * Here we reverse engineer the value of x*w, instead of transmitting it
         * directly. The used formula is the same as in the binning optimization
         * section, maybe it related to this.
         * 
         * TODO Improvement: Why not transfer beta_d * x_id?
         */
        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));

        /*
         * Compute the 1st and 2nd derivate
         * 
         * See LogRegMath.logisticFunction how we handle Numeric issues
         * 
         * TODO Bug: Why does Singh not use (xDotw + element.getXid() * betad)??
         */
        double piNew = LogRegMath.logisticFunction(
            xDotw + (element.getXid() * betad));

        batchGradient += LogRegSFOTraining.derivateL2SFO(element.getXid(),
            piNew, element.getLabel(), regularization, betad);
        batchGradientSecond += LogRegSFOTraining.derivateSecondL2SFO(
            element.getXid(), piNew, regularization);
      }

      /*
       * Newton Update
       * 
       * If first and/or second derivate is zero we probably already converged
       * to the optimum and should stop (otherwise we divide by zero)
       * 
       * TODO Why does Ng use another error-function (and derivate) where we
       * divide by N?
       */
//      betad = newtonUpdate(betad, lastUpdate, )
      if (batchGradientSecond == 0) {
        lastUpdate = 0; // Avoid division by zero (NaN), no update needed
      } else {
        lastUpdate = (batchGradient / batchGradientSecond);
        betad -= lastUpdate;
      }

      /*
       * Check for conversion
       */
      if (Math.abs(lastUpdate) < newtonTolerance) {
        converged = true;
      }

      if (dim.get() == DEBUG_DIMENSION)
        log.debug("- DEBUG: d=" + dim.get() + " iteration=" + iteration
            + " cacheSize=" + cache.size() + " grad=" + batchGradient
            + " gradSecond=" + batchGradientSecond + " new beta_d=" + betad);
    }

    // Write trained coefficient
    context.write(dim, new DoubleWritable(betad));
  }

}
