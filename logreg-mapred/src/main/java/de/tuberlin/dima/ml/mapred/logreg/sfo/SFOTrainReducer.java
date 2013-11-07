package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.logreg.sfo.LogRegSFOTraining;

public class SFOTrainReducer extends
    Reducer<IntWritable, SFOIntermediateWritable, IntWritable, DoubleWritable> {

  private int newtonMaxIterations;
  private double newtonTolerance;
  private double regularization;

  private static final int DEBUG_DIMENSION = -1; // 8609; // 196; //10394; // 12219;
  
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

    List<SFOIntermediateWritable> cache = Lists.newArrayList();

    double betad = 0;
    int iteration = 0;
    double lastUpdate = 0;
    boolean converged = false;
    
    int numPositives = 0; int numNegatives = 0;	// Debug
//    System.out.println("TRAIN d=" + dim.get());
    while ((++iteration <= newtonMaxIterations) && !converged) {

      double batchGradient = 0;
      double batchGradientSecond = 0;
      Iterable<SFOIntermediateWritable> currentIterable = (iteration == 1 ? values : cache);
      for (SFOIntermediateWritable element : currentIterable) {

        if (iteration == 1) {
          cache.add(new SFOIntermediateWritable(element));
          numPositives += (element.getLabel()==1 ? 1 : 0);
          numNegatives += (element.getLabel()==0 ? 1 : 0);
        }

        /*
         * Here we reverse engineer the value of x*w, instead of transmitting it
         * directly. The used formula is the same as in the binning optimization
         * section, maybe it related to this.
         * 
         * TODO We could also transfer beta_d * x_id here
         */
        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));

        /*
         * Compute the 1st and 2nd derivate
         * 
         * See LogRegMath.logisticFunction how we handle Numeric issues
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
    	// logger.info
    	System.out.println("- DEBUG: d=" + dim.get() + " iteration=" + iteration
            + " cacheSize=" + cache.size() + " grad=" + batchGradient
            + " gradSecond=" + batchGradientSecond + " new beta_d=" + betad + " numPositives: " + numPositives + " numNegatives: " + numNegatives);
    }

    // Write trained coefficient
    context.write(dim, new DoubleWritable(betad));
  }

}
