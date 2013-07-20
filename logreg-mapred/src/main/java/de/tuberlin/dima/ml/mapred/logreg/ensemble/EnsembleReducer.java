package de.tuberlin.dima.ml.mapred.logreg.ensemble;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import de.tuberlin.dima.ml.mapred.writables.VectorMultiLabeledWritable;
import de.tuberlin.dima.ml.validation.OnlineAccuracy;

public class EnsembleReducer extends Reducer<IntWritable, VectorMultiLabeledWritable, IntWritable, VectorWritable> {

  private int labelDimension;
  private int numFeatures;
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(EnsembleReducer.class.getName()), 
      Level.DEBUG); 
  
  @Override
    protected void setup(Context context)
        throws IOException, InterruptedException {
      super.setup(context);
      
      labelDimension = Integer.parseInt(context.getConfiguration().get(EnsembleJob.CONF_KEY_LABEL_DIMENSION));
      numFeatures = Integer.parseInt(context.getConfiguration().get(EnsembleJob.CONF_KEY_NUM_FEATURES));
    }

  @Override
  public void reduce(IntWritable key, Iterable<VectorMultiLabeledWritable> values, Context context) throws IOException, InterruptedException {
    // TRAIN
    // Use stochastic gradient descent online learning
    // Alpha, decayExponent, and stepOffset: rate and way that the learning rate decreases
    // lambda: amount of regularization
    // learningRate: initial learning rate.
    // 
    // Other comment:
    // Usually, decayExponentand stepOffsetare used to control how the initial learning
    // rate is decreased over time; setting alpha is much less common
    OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(
        2,
        numFeatures,
        new L1());
    
    learningAlgorithm
        .alpha(1)   // 1 (skipping is bad)
        .stepOffset(1000)   // 1000
        .decayExponent(0.1) // 0.9
        .lambda(3.0e-6) // 3.0e-5
        .learningRate(15);  // 20

    OnlineAccuracy accuracy = new OnlineAccuracy(0.5);
    for (VectorMultiLabeledWritable lVec : values) {
      
      // Test prediction
      int actualTarget = (int)lVec.getLabels().get(labelDimension);
      Vector vec = lVec.getVector();
      double prediction = learningAlgorithm.classifyScalar(vec);
      accuracy.addSample(actualTarget, prediction);

      // Train
      learningAlgorithm.train(actualTarget, vec);
    }
    log.debug("ONLINE TRAINING RESULTS:");
    log.debug("Accuracy: " + accuracy.getAccuracy() + " (= " + (accuracy.getTrueNegatives() + accuracy.getTruePositives()) + " / " + accuracy.getTotal() + ")");
    learningAlgorithm.close();
    
    RandomAccessSparseVector w = new RandomAccessSparseVector(learningAlgorithm.getBeta().viewRow(0));
    context.write(key, new VectorWritable(w));
  }

}
