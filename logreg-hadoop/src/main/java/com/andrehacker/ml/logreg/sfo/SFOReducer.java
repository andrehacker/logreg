package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.logreg.LogisticRegression;
import com.andrehacker.ml.util.AdaptiveLogger;
import com.google.common.collect.Lists;

public class SFOReducer extends Reducer<IntWritable, SFOIntermediateWritable, IntWritable, VectorWritable> {
  
//  private IncrementalModel model;
  
//  private static final double PENALTY = 1d;
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      SFOJob.RUN_LOCAL_MODE, Logger.getLogger(SFOReducer.class.getName()), Level.DEBUG);
  
  private static int MAX_ITERATIONS = 5;
  
  @Override
  protected void setup(Context context)
      throws IOException, InterruptedException {
    super.setup(context);
    // TODO Read Base Model!
//    model = new IncrementalModel(SFOJob.FEATURES);
    
    // TODO Read model information
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
    
    log.debug("Reducer for d=" + key.get() + " (" + SFOJob.modelInfo.getFeatureName(key.get()) + ")");
    
    List<SFOIntermediateWritable> cache = Lists.newArrayList();
    
    // TODO Improvement: Run loop until convergence
    double betad = 0;
    int iteration = 0;
    while (!(++iteration>MAX_ITERATIONS)){
      
      double batchGradient = 0;
      double batchGradientSecond = 0;
      double debugSumPi = 0;
      Iterable<SFOIntermediateWritable> currentIterable = (iteration==1 ? values : cache);
      for (SFOIntermediateWritable element : currentIterable) {
        if (iteration == 1) {
          cache.add(new SFOIntermediateWritable(element));   // We need to copy the object - uses copy constructor
        }
        double xDotw = Math.log(element.getPi() / (1 - element.getPi()));
//        xDotw = element.getPi();
        
        // TODO Improvement: Why not compute and transfer beta_d * x_id ?
        // TODO Bug: Why does Singh not use (xDotw + element.getXid() * betad)?? This is the general derivation!
        double piNew = LogisticRegression.logisticFunction(xDotw + (element.getXid() * betad));
        debugSumPi += piNew;
        batchGradient += LogisticRegression.computePartialGradientSFO(element.getXid(), piNew, element.getLabel());
        batchGradientSecond += LogisticRegression.computeSecondPartialGradientSFO(element.getXid(), piNew);
      }
      
      // TODO Apply penalty. Probably we have to divide by true N (not only non-zeros)
//      double penaltyDivN = PENALTY / cache.size();
//      firstDeri *= penaltyDivN;
//      secondDeri += penaltyDivN;
      
      // Newton update
      double update = (batchGradient / batchGradientSecond);
      betad -= update;
      
      if (key.get() == 0)
        log.debug("- it " + iteration + ": grad: " + batchGradient + " gradSecond: " + batchGradientSecond + " new beta_d: " + betad + " sumPi: " + debugSumPi);
    }   // while (!trainingDone)
    
    log.debug("- Processed " + cache.size() + " records, new beta: " + betad);
  }

}
