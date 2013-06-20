package com.andrehacker.ml.sfo;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.logreg.LogisticRegression;

public class SFOMapper extends Mapper<IntWritable, VectorWritable, IntWritable, SFOIntermediateWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static SFOIntermediateWritable outputValue = new SFOIntermediateWritable();
  
  private LogisticRegression logreg = new LogisticRegression();
  
  private IncrementalModel model;
  
//  private static AdaptiveLogger log = new AdaptiveLogger(
//      SFOJob.RUN_LOCAL_MODE, Logger.getLogger(SFOMapper.class.getName()), Level.DEBUG); 
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    // TODO Read Base Model!
    model = new IncrementalModel(SFOJob.FEATURES);
  }
  
  
  @Override
  public void map(IntWritable y, VectorWritable xi, Context context) throws IOException, InterruptedException {

    // Compute prediction for current x_i using the base model
    // TODO Improvement: Why not just compute and transmit beta * x_i ??
    double pi = logreg.predict(xi.get(), model.getW(), SFOJob.INTERCEPT);
    
    // Go through all features in x, emit value.
    // TODO Improvement: Why always transmit p_i ?? Because we need p_i for the specific x_i in reducer for each feature! Still lot of redundant traffic...
    int count=0;
    for (Vector.Element feature : xi.get().nonZeroes()) {
      // New feature?
      if (! model.getUsedDimensions().contains(feature.get())) {
        outputKey.set(feature.index());
        outputValue.setLabel(y.get());
        outputValue.setXid(feature.get());
        outputValue.setPi(pi);
        context.write(outputKey, outputValue);
        ++count;
      }
    }
    System.out.println("Mapper done. Processed " + count + " features. Basemodel prediction (pi): " + pi);
  }
}