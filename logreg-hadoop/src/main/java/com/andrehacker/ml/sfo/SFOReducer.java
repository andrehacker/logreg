package com.andrehacker.ml.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.util.AdaptiveLogger;
import com.google.common.collect.Lists;

public class SFOReducer extends Reducer<IntWritable, SFOIntermediateWritable, IntWritable, VectorWritable> {
  
  private IncrementalModel model;
  
  List<String> predictorNames = Lists.newArrayList(new String[] {
      "x", "y", "shape", "color", "xx", "xy", "yy", "a", "b", "c"
   });
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      SFOJob.RUN_LOCAL_MODE, Logger.getLogger(SFOReducer.class.getName()), Level.DEBUG); 

  @Override
  protected void setup(Context context)
      throws IOException, InterruptedException {
    super.setup(context);
    // TODO Read Base Model!
    model = new IncrementalModel(SFOJob.FEATURES);
  }

  /**
   * Notes: To iterate multiple times over data, we cache all data on heap
   * For each dimension, this stores as many objects in memory as documents have this word.
   * Assumption: Should always fit into memory
   * 
   * We could avoid this by learning with Online SGD in one pass, but for many dimensions we will have few data and this will be probably hard.
   */
  @Override
  public void reduce(IntWritable key, Iterable<SFOIntermediateWritable> values, Context context) throws IOException, InterruptedException {
    // TODO Bug: Loop over all items until convergence!
    double betad = 0;
    double delta = 0;
    double firstDeri = 0;
    double secondDeri = 0;
    
    log.debug("Reducer for d=" + key.get() + " (" + predictorNames.get(key.get()) + ")");
    
    List<SFOIntermediateWritable> cache = Lists.newArrayList();
    
    boolean trainingDone = false;
    int iteration = 1;
    while (!trainingDone) {
      firstDeri = 0;
      secondDeri = 0;
      Iterable<SFOIntermediateWritable> currentIterable = (iteration==1 ? values : cache);
      for (SFOIntermediateWritable element : currentIterable) {
        if (iteration == 1) {
          cache.add(element);
        }
        
        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));
        
        // TODO: Why not compute beta_d * x_id ?
        double piNew = logisticFunction(xDotw + betad);

        firstDeri += ((element.getLabel() - piNew) * element.getXid());
        secondDeri -= (piNew * (1 - piNew) * Math.pow(element.getXid(), 2d));
      }
      // Newton update
      delta = (firstDeri / secondDeri);
      betad -= delta;
      if (iteration==30) trainingDone = true;
      
      log.debug("- it " + iteration + ": update: " + delta + " new beta_d: " + betad);
      ++iteration;
    }
    log.debug("- Processed " + cache.size() + " records");
//    context.write(key, new VectorWritable(learningAlgorithm.getBeta().viewRow(0)));

  }

  // TODO Refactoring: Outsource this to Math helper
  private double logisticFunction(double exponent) {
    // Computes the prediction, using our current hypothesis (logistic function)
    // Overflow detection
    double negativeExp = Math.exp(-exponent);
    if (exponent != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + exponent + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }

}
