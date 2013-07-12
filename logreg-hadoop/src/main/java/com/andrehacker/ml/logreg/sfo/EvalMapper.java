package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.GlobalSettings;
import com.andrehacker.ml.logreg.LogRegMath;
import com.andrehacker.ml.writables.DoublePairWritable;

public class EvalMapper extends Mapper<IntWritable, VectorWritable, IntWritable, DoublePairWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static DoublePairWritable outputValue = new DoublePairWritable();
  
  private int numFeatures;
  private String trainOutputPath;
  
  private IncrementalModel baseModel;
  
  List<Double> coefficients;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    numFeatures = Integer.parseInt(context.getConfiguration().get(EvalJob.CONF_KEY_NUM_FEATURES));
    trainOutputPath = context.getConfiguration().get(EvalJob.CONF_KEY_TRAIN_OUTPUT);
    
    baseModel = SFOTools.readBaseModel(context.getConfiguration());
    
    coefficients = SFOTools.readTrainedCoefficients(context.getConfiguration(), numFeatures, trainOutputPath);
  }
  
  @Override
  public void map(IntWritable y, VectorWritable xi, Context context) throws IOException, InterruptedException {

    // Emit log-likelihood for new and old model (not prediction as in sfo-paper)
    // See SFOJob comments for description
    // 1) Compute log-likelihood for current x_i using the base model (without new coefficient)
    // 2) Compute log-likelihood for all unused features in xi using the related new models
    double piBase = LogRegMath.predict(xi.get(), baseModel.getW(), GlobalSettings.INTERCEPT);
    double llBase = LogRegMath.logLikelihood(y.get(), piBase); 
      // New feature?
    for (Vector.Element feature : xi.get().nonZeroes()) {
      int dim = feature.index();
      if (! baseModel.isFeatureUsed(dim)) {
        baseModel.getW().set(dim, coefficients.get(dim));
        double piNew = LogRegMath.logisticFunction(xi.get().dot(baseModel.getW()) + GlobalSettings.INTERCEPT);
        baseModel.getW().set(dim, 0d);    // reset to base model

        double llNew = LogRegMath.logLikelihood(y.get(), piNew);
        
        outputKey.set(feature.index());
//        outputValue.setFirst(piBase);
//        outputValue.setSecond(piNew);
        outputValue.setFirst(llBase);
        outputValue.setSecond(llNew);
        context.write(outputKey, outputValue);
      }
    }
  }
}