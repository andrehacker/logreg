package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.inputreader.LibSvmVectorReader;
import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFOGlobalSettings;

public class SFOTrainMapper extends Mapper<LongWritable, Text, IntWritable, SFOIntermediateWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static SFOIntermediateWritable outputValue = new SFOIntermediateWritable();
  
  private IncrementalModel baseModel;

  private boolean isMultilabelInput;
  private int positiveClass;
  private int numFeatures;
  private boolean collectDatasetStats;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    this.isMultilabelInput = Boolean.parseBoolean(context.getConfiguration().get(SFOTrainJob.CONF_KEY_IS_MULTILABEL_INPUT));
    this.positiveClass =  Integer.parseInt(context.getConfiguration().get(SFOTrainJob.CONF_KEY_POSITIVE_CLASS));
    this.numFeatures = Integer.parseInt(context.getConfiguration().get(SFOTrainJob.CONF_KEY_NUM_FEATURES));
    this.collectDatasetStats = Boolean.parseBoolean(context.getConfiguration().get(SFOEvalJob.CONF_KEY_COLLECT_DATASET_STATS));
    
    // Read base model from Distributed cache
    baseModel = SFOToolsHadoop.readBaseModelFromDC(context);
  }
  
  @Override
  public void map(LongWritable ignore, Text line, Context context) throws IOException, InterruptedException {
    // public void map(IntWritable y, VectorWritable xi, Context context) throws IOException, InterruptedException {
    
    Vector xi = new RandomAccessSparseVector(numFeatures);
    int y;
    if (isMultilabelInput) {
      y = LibSvmVectorReader.readVectorMultiLabel(xi, line.toString(), positiveClass);
    } else {
      y = LibSvmVectorReader.readVectorSingleLabel(xi, line.toString());
    }

    // Compute prediction for current x_i using the base model
    // We could also emit x * w instead - then we don't have to reverse engineer on the receiver side
    double pi = LogRegMath.predict(xi, baseModel.getW(), SFOGlobalSettings.INTERCEPT);
//    double xDotW = xi.get().dot(model.getW()) + SFOJob.INTERCEPT;
    
    for (Vector.Element feature : xi.nonZeroes()) {
      // New feature?
      if (! baseModel.isFeatureUsed(feature.index())) {
        outputKey.set(feature.index());
        outputValue.setLabel(y);
        outputValue.setXid(feature.get());
        outputValue.setPi(pi);
//        outputValue.setPi(xDotW);
        context.write(outputKey, outputValue);
      }
      if (collectDatasetStats) {
    	context.getCounter(SFOEvalJob.SFO_EVAL_COUNTER.NUM_NON_ZEROS).increment(1);
      }
    }
  }
}