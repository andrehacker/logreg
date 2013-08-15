package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import edu.stanford.nlp.optimization.DiffFunction;

/**
 * Logistic Regression objective function for QNMinimizer.
 * 
 * Computation of valueAt(double[]) and derivativeAt(double[]) spawn MR jobs.
 */
public class LBFGSDiffFunction implements DiffFunction {

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      Logger.getLogger(LBFGSDiffFunction.class.getName()), 
      Level.DEBUG);

  private TrainingErrorJob trainingErrorJob;
  private GradientJob gradientJob;

  public LBFGSDiffFunction(TrainingErrorJob trainingErrorJob, GradientJob gradientJob) {
    this.trainingErrorJob = trainingErrorJob;
    this.gradientJob = gradientJob;
  }

  public static HashMap<Integer, Integer> count = new HashMap<Integer, Integer>();

  @Override
  public double valueAt(double[] weights) {

    this.trainingErrorJob.setWeightVector(weights);

    // run job
    try {
      ToolRunner.run(this.trainingErrorJob, null);
      LOGGER.debug("valueAt: completed MR job");
    } catch (Exception e1) {
      e1.printStackTrace();
    }

    // return output
    try {
      FileSystem fs = FileSystem.get(this.trainingErrorJob.getConf());
      Path outputPath = new Path(this.trainingErrorJob.getOutputPath());
      FileStatus[] output = fs.listStatus(outputPath, new IterationOutputFilter());

      for (Pair<NullWritable, DoubleWritable> values : new SequenceFileIterable<NullWritable, DoubleWritable>(
          output[0].getPath(), fs.getConf())) {

        double trainingError = values.getSecond().get();

        LOGGER.debug("valueAt: read trainingError from MR job: " + trainingError);

        return trainingError;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    throw new RuntimeException();
  }

  @Override
  public int domainDimension() {
    // TODO Add a parameter for numFeatures
    return 47237;
  }

  @Override
  public double[] derivativeAt(double[] weights) {

    this.gradientJob.setWeightVector(weights);

    // run job
    try {
      ToolRunner.run(this.gradientJob, null);
      LOGGER.debug("derivativeAt: completed MR job");
    } catch (Exception e1) {
      e1.printStackTrace();
    }

    // return output
    try {
      FileSystem fs = FileSystem.get(this.gradientJob.getConf());
      Path outputPath = new Path(this.gradientJob.getOutputPath());
      FileStatus[] output = fs.listStatus(outputPath, new IterationOutputFilter());

      for (Pair<NullWritable, VectorWritable> weightVector : new SequenceFileIterable<NullWritable, VectorWritable>(
          output[0].getPath(), fs.getConf())) {

        Vector gradient = weightVector.getSecond().get();
        LOGGER.debug("derivativeAt: read gradient from MR job: " + gradient);

        // Vector to double array
        double[] d = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
          d[i] = gradient.get(i);
        }

        return d;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    throw new RuntimeException();
  }

  private static class IterationOutputFilter implements PathFilter {
    @Override
    public boolean accept(Path path) {
      if (path.getName().startsWith("part"))
        return true;

      return false;
    }
  }
}