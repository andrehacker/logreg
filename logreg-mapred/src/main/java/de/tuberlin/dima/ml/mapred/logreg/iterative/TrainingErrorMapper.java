package de.tuberlin.dima.ml.mapred.logreg.iterative;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.mapred.writables.IDAndLabels;

public class TrainingErrorMapper extends
    Mapper<IDAndLabels, VectorWritable, NullWritable, DoubleWritable> {
  
  private int labelDimension;

  private Vector w;
  private DoubleWritable trainingError = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    labelDimension = Integer.parseInt(context.getConfiguration().get(TrainingErrorJob.CONF_KEY_LABEL_DIMENSION));

    // read initial weights
    Configuration conf = context.getConfiguration();
    Path[] iterationWeights = DistributedCache.getLocalCacheFiles(conf);

    if (iterationWeights == null) { throw new RuntimeException("No weights set"); }

    Path localPath = new Path("file://" + iterationWeights[0].toString());

    for (Pair<NullWritable, VectorWritable> weights : new SequenceFileIterable<NullWritable, VectorWritable>(
        localPath, conf)) {

      this.w = weights.getSecond().get();
    }
  }

  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException,
      InterruptedException {

    Vector x = value.get();
    double y = key.getLabels().get(labelDimension);

    this.trainingError.set(LogRegMath.computeSqError(x, this.w, y));

    context.write(NullWritable.get(), this.trainingError);
  }
}