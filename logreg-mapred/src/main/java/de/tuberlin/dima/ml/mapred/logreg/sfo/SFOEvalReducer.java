package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import de.tuberlin.dima.ml.mapred.GlobalSettings;
import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import de.tuberlin.dima.ml.writables.DoublePairWritable;

public class SFOEvalReducer extends Reducer<IntWritable, DoublePairWritable, IntWritable, DoubleWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(SFOEvalReducer.class.getName()), GlobalSettings.LOG_LEVEL);
  
  @Override
  public void reduce(IntWritable dim, Iterable<DoublePairWritable> values, Context context) throws IOException, InterruptedException {
    
    log.debug("Eval Reducer for d=" + dim.get());
    
    int count=0;
    double sumLlBase=0;
    double sumLlNew=0;
    for (DoublePairWritable element : values) {
      sumLlBase += element.getFirst();
      sumLlNew += element.getSecond();
      ++count;
    }
    double gain = sumLlNew - sumLlBase;
    context.write(dim, new DoubleWritable(gain));

    log.debug("- ll_base: " + sumLlBase + " ll_new: " + sumLlNew + " GAIN: " + gain);
    log.debug("- Processed " + count + " records");
  }
}
