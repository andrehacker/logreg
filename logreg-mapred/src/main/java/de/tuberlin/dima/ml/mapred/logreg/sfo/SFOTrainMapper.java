package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFOGlobalSettings;

public class SFOTrainMapper extends Mapper<IntWritable, VectorWritable, IntWritable, SFOIntermediateWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static SFOIntermediateWritable outputValue = new SFOIntermediateWritable();
  private static final Log LOG = LogFactory.getLog(SFOTrainMapper.class);
  
  private IncrementalModel baseModel;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    baseModel = SFOToolsHadoop.readBaseModel(context.getConfiguration());
    System.out.println("STDOUT: Setup Train Mapper");
    LOG.info("COMMONS_LOGGING: Setup Train Mapper");
  }
  
  
  @Override
  public void map(IntWritable y, VectorWritable xi, Context context) throws IOException, InterruptedException {

    // Compute prediction for current x_i using the base model
    // TODO Improvement: Why not just compute and transmit beta * x_i ??
    double pi = LogRegMath.predict(xi.get(), baseModel.getW(), SFOGlobalSettings.INTERCEPT);
//    double xDotW = xi.get().dot(model.getW()) + SFOJob.INTERCEPT;
    
    // Go through all features in x, emit value.
    // TODO Improvement: Why always transmit p_i ?? Because we need p_i for the specific x_i in reducer for each feature! Still lot of redundant traffic...
    for (Vector.Element feature : xi.get().nonZeroes()) {
      // New feature?
      if (! baseModel.isFeatureUsed(feature.index())) {
        outputKey.set(feature.index());
        outputValue.setLabel(y.get());
        outputValue.setXid(feature.get());
        outputValue.setPi(pi);
//        outputValue.setPi(xDotW);
        context.write(outputKey, outputValue);
      }
    }
  }
}