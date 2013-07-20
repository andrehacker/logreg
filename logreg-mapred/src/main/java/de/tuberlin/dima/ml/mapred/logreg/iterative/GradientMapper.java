package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import de.tuberlin.dima.ml.logreg.LogRegModel;
import de.tuberlin.dima.ml.mapred.util.AdaptiveLogger;
import de.tuberlin.dima.ml.mapred.writables.IDAndLabels;

public class GradientMapper extends Mapper<IDAndLabels, VectorWritable, NullWritable, VectorWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(GradientMapper.class.getName()), 
      Level.DEBUG);
  
  private int labelDimension;
  private int numFeatures;
  
  private LogRegModel logreg;
  
  private Vector batchGradient;
  private long count=0;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    labelDimension = Integer.parseInt(context.getConfiguration().get(GradientJob.CONF_KEY_LABEL_DIMENSION));
    numFeatures = Integer.parseInt(context.getConfiguration().get(GradientJob.CONF_KEY_NUM_FEATURES));
    
    batchGradient = new RandomAccessSparseVector(numFeatures);

    Configuration conf = context.getConfiguration();
    Path[] iterationWeights = DistributedCache.getLocalCacheFiles(conf);

    if (iterationWeights == null) { throw new RuntimeException("No weights set"); }

    Path localPath = new Path("file://" + iterationWeights[0].toString());

    Vector w = null;
    for (Pair<NullWritable, VectorWritable> weights : new SequenceFileIterable<NullWritable, VectorWritable>(
        localPath, conf)) {
      w = weights.getSecond().get();
      System.out.println("Read from distributed cache in gradient mapper");
      System.out.println("- non zeros: " + w.getNumNonZeroElements());

    }

    this.logreg = new LogRegModel(w, 0.5d);
  }

  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException, InterruptedException {
    
    // Compute gradient regarding current data point
    Vector gradient = logreg.computePartialGradient(value.get(), (int)key.getLabels().get(labelDimension));
    batchGradient.assign(gradient, Functions.PLUS);
    ++count;
  }
  
  @Override
  protected void cleanup(Context context)
      throws IOException, InterruptedException {
    super.cleanup(context);
    context.write(NullWritable.get(), new VectorWritable(batchGradient));
    log.debug("Mapper: partial sum of gradient (" + count + " items)");
    log.debug("- Dimensions: " + batchGradient.size() + " Non Zero: " + batchGradient.getNumNonZeroElements());
  }
}