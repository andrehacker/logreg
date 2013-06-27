package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.andrehacker.ml.util.AdaptiveLogger;
import com.andrehacker.ml.writables.DoublePairWritable;

public class EvalReducer extends Reducer<IntWritable, DoublePairWritable, IntWritable, DoubleWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      SFOJob.RUN_LOCAL_MODE, Logger.getLogger(EvalReducer.class.getName()), Level.DEBUG);
  
  @Override
  public void reduce(IntWritable dim, Iterable<DoublePairWritable> values, Context context) throws IOException, InterruptedException {
    
    log.debug("Eval Reducer for d=" + dim.get() + " (" + SFOJob.modelInfo.getFeatureName(dim.get()) + ")");
    
    int count=0;
    double sumPiBase=0;
    double sumPiNew=0;
    for (DoublePairWritable element : values) {
//      log.debug(" - " + element.getFirst() + " " + element.getSecond());
      sumPiBase += element.getFirst();
      sumPiNew += element.getSecond();
      ++count;
    }
    double gain = sumPiNew - sumPiBase;
    context.write(dim, new DoubleWritable(gain));

    log.debug("Pi_base: " + sumPiBase + " Pi_new: " + sumPiNew + " Gain: " + gain);
    log.debug("=> Processed " + count + " records");
  }
}
