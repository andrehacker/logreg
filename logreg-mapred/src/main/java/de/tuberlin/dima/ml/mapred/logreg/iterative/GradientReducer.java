package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.io.IOException;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;

public class GradientReducer extends Reducer<NullWritable, VectorWritable, NullWritable, VectorWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(GradientReducer.class.getName()), Level.DEBUG);

  private int numFeatures;
  
  @Override
  protected void setup(Context context)
      throws IOException, InterruptedException {
    super.setup(context);

    numFeatures = Integer.parseInt(context.getConfiguration().get(GradientJob.CONF_KEY_NUM_FEATURES));
  }
  
  @Override
  public void reduce(NullWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    Vector batchGradientSum = new RandomAccessSparseVector(numFeatures);
    
    for (VectorWritable gradient : values) {
      batchGradientSum.assign(gradient.get(), Functions.PLUS);
    }
    
    log.debug("Gradient result: Dimensions: " + batchGradientSum.size() + " Non Zero: " + batchGradientSum.getNumNonZeroElements());
    context.write(NullWritable.get(), new VectorWritable(batchGradientSum));
  }
  
}
