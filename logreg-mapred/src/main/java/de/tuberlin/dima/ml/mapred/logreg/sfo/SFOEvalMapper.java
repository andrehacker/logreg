package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
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

// public class SFOEvalMapper extends Mapper<IntWritable, VectorWritable, IntWritable, DoublePairWritable> {
public class SFOEvalMapper extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static DoubleWritable outputValue = new DoubleWritable();
  
  private boolean isMultilabelInput;
  private int positiveClass;
  private int numFeatures;
  private String trainOutputPath;
  private boolean collectDatasetStats;
  
  private IncrementalModel baseModel;
  
  List<Double> coefficients;
  
  private static final int DEBUG_DIMENSION = -1; // 8609; // 196; //10394; // 12219;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    this.isMultilabelInput = Boolean.parseBoolean(context.getConfiguration().get(SFOEvalJob.CONF_KEY_IS_MULTILABEL_INPUT));
    this.positiveClass = Integer.parseInt(context.getConfiguration().get(SFOEvalJob.CONF_KEY_POSITIVE_CLASS));
    this.numFeatures = Integer.parseInt(context.getConfiguration().get(SFOEvalJob.CONF_KEY_NUM_FEATURES));
    this.trainOutputPath = context.getConfiguration().get(SFOEvalJob.CONF_KEY_TRAIN_OUTPUT);
    this.collectDatasetStats = Boolean.parseBoolean(context.getConfiguration().get(SFOEvalJob.CONF_KEY_COLLECT_DATASET_STATS));
    
    // Read base model from Distributed cache
    baseModel = SFOToolsHadoop.readBaseModelFromDC(context);
    
    // Read trained coefficients from output of previous job
    coefficients = SFOToolsHadoop.readTrainedCoefficients(context.getConfiguration(), trainOutputPath, numFeatures);
  }
  
  @Override
  public void map(LongWritable ignore, Text line, Context context) throws IOException, InterruptedException {
	
    Vector xi = new RandomAccessSparseVector(numFeatures);
    int y;
    if (isMultilabelInput) {
      y = LibSvmVectorReader.readVectorMultiLabel(xi, line.toString(), positiveClass);
    } else {
      y = LibSvmVectorReader.readVectorSingleLabel(xi, line.toString());
    }
    
    // Compute log-likelihood for current x_i using the base model (without new coefficient)
    double piBase = LogRegMath.predict(xi, baseModel.getW(), SFOGlobalSettings.INTERCEPT);
    double llBase = LogRegMath.logLikelihood(y, piBase);
    
    // Compute log-likelihood for all unused features in xi using the newly trained
    // approximate models
    for (Vector.Element feature : xi.nonZeroes()) {
      int dim = feature.index();
      if (! baseModel.isFeatureUsed(dim)) {
    	Double coefficient = coefficients.get(dim);
    	// Features we did not have in our training data won't have a coefficient 
    	if (coefficient != null) {
          baseModel.getW().set(dim, coefficient);
          double piNew = LogRegMath.logisticFunction(xi.dot(baseModel.getW()) + SFOGlobalSettings.INTERCEPT);
          baseModel.getW().set(dim, 0d);    // reset to base model
  
          double llNew = LogRegMath.logLikelihood(y, piNew);
          
          outputKey.set(feature.index());
          outputValue.set(llNew - llBase);
          context.write(outputKey, outputValue);

          if (feature.index() == DEBUG_DIMENSION) {
        	System.out.println(feature.index() + ": PI BASE: " + piBase + " LLBASE: " + llBase + " y: " + y);
        	System.out.println(feature.index() + ": > PI NEW: " + piNew + " LLNEW: " + llNew);
          }
        }
      }
      if (collectDatasetStats) {
    	context.getCounter(SFOEvalJob.SFO_EVAL_COUNTER.NUM_NON_ZEROS).increment(1);
      }
    }
  }
}