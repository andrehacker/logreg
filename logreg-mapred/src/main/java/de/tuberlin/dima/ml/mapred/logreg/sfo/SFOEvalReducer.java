package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

public class SFOEvalReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
  
//  private static AdaptiveLogger log = new AdaptiveLogger(
//      Logger.getLogger(SFOEvalReducer.class.getName()), GlobalSettings.LOG_LEVEL);
  
  private static final int DEBUG_DIMENSION = -1; // 8609; // 196; //10394; //12219
  
  @Override
  public void reduce(IntWritable dim, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
    
//    log.debug("Eval Reducer for d=" + dim.get());
    
    double gain=0;
    int count=0;
    for (DoubleWritable element : values) {
      gain += element.get();

      if (dim.get() == DEBUG_DIMENSION) {
        System.out.println("- gain += " + element.get());
      }
      ++count;
    }
    context.write(dim, new DoubleWritable(gain));

    if (dim.get() == DEBUG_DIMENSION) {
      System.out.println("Eval for d " + dim.get() + ": count: " + count + " GAIN: " + gain);
    }
  }
}
