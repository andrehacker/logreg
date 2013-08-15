package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.io.File;
import java.io.FileFilter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.base.Joiner;

import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import de.tuberlin.dima.ml.mapred.util.HadoopUtils;

/**
 * NOT-FINISHED Batch gradient for logistic regression
 * TODO Apply gradient to current model instead of overwriting it
 */
public class BatchGradientDriver {

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      Logger.getLogger(BatchGradientDriver.class.getName()), 
      Level.DEBUG);

  private String inputFile;
  private String outputPath;
  private final int maxIterations;
  private int labelDimension;
  private int numFeatures;
  private String hdfsAddress;
  private String jobTrackerAddress;

  private final VectorWritable weights;

  private static final Joiner pathJoiner = Joiner.on("/");

  public BatchGradientDriver(
      String inputFile,
      String outputPath,
      int maxIterations,
      double initial,
      int labelDimension,
      int numFeatures,
      String hdfsAddress,
      String jobTrackerAddress) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.labelDimension = labelDimension;
    this.numFeatures = numFeatures;
    this.hdfsAddress = hdfsAddress;
    this.jobTrackerAddress = jobTrackerAddress;

    this.maxIterations = maxIterations;

    Vector vec = new SequentialAccessSparseVector(numFeatures);
    
    vec.assign(initial);

    this.weights = new VectorWritable(vec);
  }
  
  public int train() throws Exception {

    // Non zero numbers for rcv1-v2 (5000): 21871 -> 19199 -> 19165
    
    boolean[] hasSucceeded = new boolean[this.maxIterations];
    
    // Configuration object for file system actions
    Configuration conf = HadoopUtils.createConfiguration(hdfsAddress, jobTrackerAddress);
    boolean runLocal = HadoopUtils.detectLocalMode(conf);

    // iterations
    for (int i = 0; i < this.maxIterations; i++) {
      LOGGER.debug("> starting iteration " + i);
      
      // output path for this iteration
      Path iterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + i));

      FileSystem fs = FileSystem.get(conf);
      
      GradientJob job = new GradientJob(
          inputFile,
          iterationPath.toString(),
          labelDimension,
          numFeatures);
      
      if (i == 0) {

        // Remove data from previous runs (delete root output folder recursively)
        if (runLocal) {
          new DeletingVisitor().accept(new File(this.outputPath));
        } else {
          fs.delete(new Path(this.outputPath), true);
        }
        
      } else {

        // Add weights of previous iteration to DistributedCache (existing file)
        Path prevIterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + (i - 1)));
        FileStatus[] prevIterationWeights = fs.listStatus(prevIterationPath, new IterationOutputFilter());
        
        Path cachePath = prevIterationWeights[0].getPath();
        this.weights.set(readVectorFromHDFS(cachePath, conf));

      }
      
      // GradientJob will write this vector to distributed cache
      job.setWeightVector(this.weights.get());
      
      // execute job
      hasSucceeded[i] = (ToolRunner.run(conf, job, null)==0) ? true : false;
      LOGGER.debug("> completed iteration? " + hasSucceeded[i]);
    }

    for (int i = 0; i < this.maxIterations; i++) {
      if (!hasSucceeded[i])
        return 1;
    }

    return 0;
  }
  
  private Vector readVectorFromHDFS(Path filePath, Configuration conf) {
    Vector w = null;
    for (Pair<NullWritable, VectorWritable> weights : new SequenceFileIterable<NullWritable, VectorWritable>(
        filePath, conf)) {
      w = weights.getSecond().get();
      System.out.println("Read from distributed cache in gradient mapper");
      System.out.println("- non zeros: " + w.getNumNonZeroElements());
    }
    return w;
  }

  private static class IterationOutputFilter implements PathFilter {

    @Override
    public boolean accept(Path path) {
      if (path.getName().startsWith("part"))
        return true;

      return false;
    }
  }

  /**
   * Copied from MahoutTestCase. Recursively deletes folder and contained files
   */
  private static class DeletingVisitor implements FileFilter {

    @Override
    public boolean accept(File f) {
      if (!f.isFile()) {
        f.listFiles(this);
      }
      f.delete();
      return false;
    }
  }
}