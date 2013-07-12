package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import com.andrehacker.ml.GlobalSettings;
import com.andrehacker.ml.util.AdaptiveLogger;
import com.andrehacker.ml.writables.DoublePairWritable;

public class EvalReducer extends Reducer<IntWritable, DoublePairWritable, IntWritable, DoubleWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(EvalReducer.class.getName()), GlobalSettings.LOG_LEVEL);
  
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
